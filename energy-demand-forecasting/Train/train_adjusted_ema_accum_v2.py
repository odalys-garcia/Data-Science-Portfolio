from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from generate_dataset_skipbad_debugged import (
        EnergyWeatherDataset,
        compute_normalization_stats,
        build_or_load_weather_manifest,
        prebuild_manifests,
        resolve_selected_idx,
        safe_load_weather,
    )
except ImportError:
    try:
        from generate_dataset_skipbad import (
            EnergyWeatherDataset,
            compute_normalization_stats,
            build_or_load_weather_manifest,
            prebuild_manifests,
            resolve_selected_idx,
            safe_load_weather,
        )
    except ImportError:
        from generate_dataset import EnergyWeatherDataset, compute_normalization_stats  # type: ignore
        build_or_load_weather_manifest = None
        prebuild_manifests = None
        resolve_selected_idx = None
        safe_load_weather = None

try:
    from model_adjusted_v2 import get_model
except ImportError:
    try:
        from model_adjusted_skipbad_debugged import get_model
    except ImportError:
        try:
            from model_adjusted_skipbad import get_model
        except ImportError:
            try:
                from model_adjusted import get_model
            except ImportError:
                from model import get_model  # type: ignore

# ============================================================
# Configuration
# ============================================================
WEATHER_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data"
ENERGY_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data"
TRAIN_YEARS = [2019, 2020, 2021, 2022]
VAL_YEARS = [2023]

SELECTED_VARS: List[str] = []
SEQ_LEN_HIST = 24
SEQ_LEN_FUT = 24

BATCH_SIZE = 4
EPOCHS = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = torch.cuda.is_available()
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR = SCRIPT_DIR / "checkpoints"
DEFAULT_MODEL_NAME = "adjusted_hybrid"
MANIFEST_DIR = SCRIPT_DIR / "manifests"
USE_MANIFEST = True
REBUILD_MANIFEST = False
PREBUILD_MANIFESTS = True

WEATHER_STATS_STRIDE = 6
WEATHER_STATS_MAX_FILES = None
WEATHER_STD_FLOOR = 0.5
WEATHER_NORM_CLIP = 10.0

MODEL_CONFIG = {
    "seq_len_hist": SEQ_LEN_HIST,
    "seq_len_fut": SEQ_LEN_FUT,
    "weather_dim": 128,
    "tab_dim": 128,
    "decoder_dim": 128,
    "grid_size": (4, 4),
    "num_future_layers": 2,
    "num_heads": 4,
    "head_hidden": 128,
    "dropout": 0.10,
}


def _build_runtime_model_config(seq_len_hist: int, seq_len_fut: int) -> dict:
    return {
        "seq_len_hist": int(seq_len_hist),
        "seq_len_fut": int(seq_len_fut),
        "weather_dim": int(MODEL_CONFIG["weather_dim"]),
        "tab_dim": int(MODEL_CONFIG["tab_dim"]),
        "decoder_dim": int(MODEL_CONFIG["decoder_dim"]),
        "grid_size": tuple(MODEL_CONFIG["grid_size"]),
        "num_future_layers": int(MODEL_CONFIG["num_future_layers"]),
        "num_heads": int(MODEL_CONFIG["num_heads"]),
        "head_hidden": int(MODEL_CONFIG["head_hidden"]),
        "dropout": float(MODEL_CONFIG["dropout"]),
    }

def _sanitize_model_name(model_name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(model_name).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or DEFAULT_MODEL_NAME


def build_output_paths(model_name: str) -> dict:
    safe_model_name = _sanitize_model_name(model_name)
    return {
        "model_name": safe_model_name,
        "primary_checkpoint_path": SCRIPT_DIR / f"{safe_model_name}.pth",
        "latest_checkpoint_path": SAVE_DIR / f"{safe_model_name}_latest.pth",
        "backup_best_path": SAVE_DIR / f"{safe_model_name}_best.pth",
        "history_path": SAVE_DIR / f"{safe_model_name}_training_history.csv",
    }


def calculate_mape(preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    abs_diff = torch.abs(targets - preds)
    denom = torch.clamp(torch.abs(targets), min=epsilon)
    return (abs_diff / denom).mean() * 100.0


def unnormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.view(1, 1, -1).to(tensor.device)
    std = std.view(1, 1, -1).to(tensor.device)
    return tensor * std + mean


def tensors_all_finite(*tensors: torch.Tensor) -> bool:
    return all(torch.isfinite(t).all() for t in tensors)


def tensor_stats_string(name: str, tensor: torch.Tensor) -> str:
    t = tensor.detach().float().cpu()
    finite = torch.isfinite(t)
    finite_vals = t[finite]
    summary = {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "nan_count": int(torch.isnan(t).sum().item()),
        "posinf_count": int(torch.isposinf(t).sum().item()),
        "neginf_count": int(torch.isneginf(t).sum().item()),
    }
    if finite_vals.numel() > 0:
        summary.update({
            "finite_min": float(finite_vals.min().item()),
            "finite_max": float(finite_vals.max().item()),
            "finite_mean": float(finite_vals.mean().item()),
            "finite_std": float(finite_vals.std(unbiased=False).item()),
        })
    else:
        summary.update({
            "finite_min": None,
            "finite_max": None,
            "finite_mean": None,
            "finite_std": None,
        })
    return str(summary)


def raise_nonfinite_batch_error(stage: str, amp_enabled: bool, tensors: Sequence[Tuple[str, torch.Tensor]]) -> None:
    lines = [f"Non-finite tensors detected during {stage}. amp_enabled={amp_enabled}"]
    for name, tensor in tensors:
        lines.append(tensor_stats_string(name, tensor))
    raise RuntimeError("\n".join(lines))


def build_weather_file_list(weather_dir: Path, years: Sequence[int]) -> List[Path]:
    files: List[Path] = []
    for year in sorted(set(int(y) for y in years)):
        year_dir = weather_dir / str(year)
        if year_dir.is_dir():
            files.extend(sorted(year_dir.glob("X_*.pt")))
    return files


def compute_weather_channel_stats(
    weather_dir: Path,
    years: Sequence[int],
    stride: int = 6,
    max_files: Optional[int] = None,
    manifest_dir: Optional[Path] = None,
    selected_vars: Optional[Sequence[str]] = None,
    rebuild_manifest: bool = False,
    std_floor: float = WEATHER_STD_FLOOR,
) -> Tuple[torch.Tensor, torch.Tensor]:
    file_list = build_weather_file_list(weather_dir, years)
    if len(file_list) == 0:
        raise FileNotFoundError(f"No weather files found in {weather_dir} for years {years}")

    selected_vars_list = list(selected_vars or [])
    selected_idx = resolve_selected_idx(weather_dir, selected_vars_list) if resolve_selected_idx is not None else None

    finite_keys = None
    if build_or_load_weather_manifest is not None:
        manifest, _ = build_or_load_weather_manifest(
            weather_dir=weather_dir,
            years=years,
            selected_vars=selected_vars_list,
            selected_idx=selected_idx,
            manifest_dir=manifest_dir,
            rebuild_manifest=rebuild_manifest,
            verbose=False,
        )
        finite_keys = {k for k, v in manifest.get("files", {}).items() if bool(v.get("is_finite", False))}

    selected = []
    for i, fpath in enumerate(file_list):
        if stride > 1 and (i % stride) != 0:
            continue
        key = fpath.stem[2:]
        if finite_keys is not None and key not in finite_keys:
            continue
        selected.append(fpath)
        if max_files is not None and len(selected) >= max_files:
            break

    sum_channels = None
    sumsq_channels = None
    count = 0
    skipped_nonfinite = 0
    skipped_invalid_shape = 0

    print(f"Estimating weather normalization stats from {len(selected)} sampled tensors ...")
    for fpath in tqdm(selected, desc="Weather stats"):
        if safe_load_weather is not None:
            x = safe_load_weather(fpath)
        else:
            try:
                x = torch.load(fpath, map_location="cpu", weights_only=True)
            except TypeError:
                x = torch.load(fpath, map_location="cpu")
        if selected_idx is not None:
            x = x[..., selected_idx]
        x = x.float()
        if x.ndim != 3:
            skipped_invalid_shape += 1
            continue
        if not torch.isfinite(x).all():
            skipped_nonfinite += 1
            continue
        flat = x.reshape(-1, x.shape[-1]).double()
        channel_sum = flat.sum(dim=0)
        channel_sumsq = (flat * flat).sum(dim=0)
        if sum_channels is None:
            sum_channels = channel_sum
            sumsq_channels = channel_sumsq
        else:
            sum_channels += channel_sum
            sumsq_channels += channel_sumsq
        count += flat.shape[0]

    if skipped_nonfinite > 0 or skipped_invalid_shape > 0:
        print(f"[WARNING] Weather-stat estimation skipped {skipped_nonfinite} non-finite tensors and {skipped_invalid_shape} tensors with invalid shape.")
    if count == 0 or sum_channels is None or sumsq_channels is None:
        raise RuntimeError("Weather statistics accumulation failed: no finite sampled weather tensors were available.")

    mean = sum_channels / float(count)
    var = torch.clamp((sumsq_channels / float(count)) - (mean * mean), min=0.0)
    std = torch.sqrt(var)
    std_floor = float(std_floor)
    num_floored = int((std < std_floor).sum().item())
    if num_floored > 0:
        print(f"[INFO] Applying weather std floor={std_floor:.6f} to {num_floored} channel(s).")
    std = torch.clamp(std, min=std_floor)
    return mean.float(), std.float()


def compute_training_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    horizon_weights = torch.linspace(1.0, 1.20, steps=pred_norm.size(1), device=pred_norm.device, dtype=pred_norm.dtype).view(1, -1, 1)
    huber = torch.nn.functional.smooth_l1_loss(pred_norm, target_norm, reduction="none")
    huber = (huber * horizon_weights).mean()
    pred_raw = unnormalize(pred_norm, norm_mean, norm_std)
    target_raw = unnormalize(target_norm, norm_mean, norm_std)
    mape = calculate_mape(pred_raw, target_raw) / 100.0
    total_loss = 0.80 * huber + 0.20 * mape
    return total_loss, huber.detach(), mape.detach()

def clone_tensor_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, tensor in self._named_state_tensors(model):
            self.shadow[name] = tensor.detach().clone()

    @staticmethod
    def _named_state_tensors(model: nn.Module):
        for name, param in model.named_parameters():
            yield name, param
        for name, buf in model.named_buffers():
            if buf is not None:
                yield name, buf

    @staticmethod
    def _named_model_state(model: nn.Module):
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        return params, buffers

    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        one_minus_decay = 1.0 - self.decay
        for name, tensor in self._named_state_tensors(model):
            shadow_tensor = self.shadow[name]
            source = tensor.detach()
            if shadow_tensor.is_floating_point():
                shadow_tensor.lerp_(source, one_minus_decay)
            else:
                shadow_tensor.copy_(source)

    def copy_to(self, model: nn.Module) -> None:
        params, buffers = self._named_model_state(model)
        for name, shadow_tensor in self.shadow.items():
            if name in params:
                params[name].data.copy_(shadow_tensor)
            elif name in buffers and buffers[name] is not None:
                buffers[name].data.copy_(shadow_tensor)

    def store(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in self._named_state_tensors(model)}

    def restore(self, model: nn.Module, backup_state: Dict[str, torch.Tensor]) -> None:
        params, buffers = self._named_model_state(model)
        for name, backup_tensor in backup_state.items():
            if name in params:
                params[name].data.copy_(backup_tensor)
            elif name in buffers and buffers[name] is not None:
                buffers[name].data.copy_(backup_tensor)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return clone_tensor_state_dict(self.shadow)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the adjusted hybrid model with manifest-backed dataset validity caching.")
    parser.add_argument("--weather-dir", type=str, default=WEATHER_DIR)
    parser.add_argument("--energy-dir", type=str, default=ENERGY_DIR)
    parser.add_argument("--train-years", type=str, default=",".join(map(str, TRAIN_YEARS)))
    parser.add_argument("--val-years", type=str, default=",".join(map(str, VAL_YEARS)))
    parser.add_argument("--manifest-dir", type=str, default=str(MANIFEST_DIR))
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--skip-prebuild-manifests", action="store_true")
    parser.add_argument("--seq-len-hist", type=int, default=SEQ_LEN_HIST)
    parser.add_argument("--seq-len-fut", type=int, default=SEQ_LEN_FUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--weather-std-floor", type=float, default=WEATHER_STD_FLOOR)
    parser.add_argument("--weather-norm-clip", type=float, default=WEATHER_NORM_CLIP)
    parser.add_argument("--skip-nonfinite-batches", action="store_true")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--disable-ema", action="store_true")
    return parser


def _parse_years(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    args = build_parser().parse_args()

    weather_dir = Path(args.weather_dir)
    energy_dir = Path(args.energy_dir)
    train_years = _parse_years(args.train_years)
    val_years = _parse_years(args.val_years)
    manifest_dir = Path(args.manifest_dir)
    use_manifest = not args.no_manifest
    rebuild_manifest = bool(args.rebuild_manifest)
    amp_enabled = (DEVICE.type == "cuda") and (not args.disable_amp)
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    ema_enabled = not args.disable_ema

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = build_output_paths(args.model_name)
    print(
        f"Using device: {DEVICE} | amp_enabled={amp_enabled} | "
        f"grad_accum_steps={grad_accum_steps} | ema_enabled={ema_enabled} | "
        f"model_name={output_paths['model_name']}"
    )

    if use_manifest and prebuild_manifests is not None and not args.skip_prebuild_manifests:
        print("Prebuilding manifests...")
        prebuild_manifests(
            weather_dir=weather_dir,
            energy_dir=energy_dir,
            years_groups=[train_years, val_years],
            selected_vars=SELECTED_VARS,
            seq_configs=[(args.seq_len_hist, args.seq_len_fut)],
            manifest_dir=manifest_dir,
            rebuild_manifest=rebuild_manifest,
            verbose=True,
        )

    print("Initializing training dataset...")
    train_dataset = EnergyWeatherDataset(
        weather_dir=weather_dir,
        energy_dir=energy_dir,
        years=train_years,
        selected_vars=SELECTED_VARS,
        norm_mean=None,
        norm_std=None,
        scan_workers=8,
        seq_len_hist=args.seq_len_hist,
        seq_len_fut=args.seq_len_fut,
        manifest_dir=manifest_dir,
        use_manifest=use_manifest,
        rebuild_manifest=rebuild_manifest,
        verbose=True,
    )

    train_targets_df = pd.DataFrame(train_dataset._targets_raw.numpy(), columns=train_dataset.zone_names)
    norm_mean, norm_std = compute_normalization_stats(train_targets_df)
    train_dataset.norm_mean = norm_mean
    train_dataset.norm_std = norm_std

    print("Initializing validation dataset...")
    val_dataset = EnergyWeatherDataset(
        weather_dir=weather_dir,
        energy_dir=energy_dir,
        years=val_years,
        selected_vars=SELECTED_VARS,
        norm_mean=norm_mean,
        norm_std=norm_std,
        scan_workers=8,
        seq_len_hist=args.seq_len_hist,
        seq_len_fut=args.seq_len_fut,
        manifest_dir=manifest_dir,
        use_manifest=use_manifest,
        rebuild_manifest=rebuild_manifest,
        verbose=True,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"Dataset is empty after validity filtering: train_windows={len(train_dataset)}, val_windows={len(val_dataset)}"
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=PIN_MEMORY, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=PIN_MEMORY, drop_last=False)

    sample_weather_seq = train_dataset[0][0]
    actual_in_channels = int(sample_weather_seq.shape[1])
    print(f"Detected {actual_in_channels} weather channels.")

    weather_mean, weather_std = compute_weather_channel_stats(
        weather_dir=weather_dir,
        years=train_years,
        stride=WEATHER_STATS_STRIDE,
        max_files=WEATHER_STATS_MAX_FILES,
        manifest_dir=manifest_dir,
        selected_vars=SELECTED_VARS,
        rebuild_manifest=rebuild_manifest,
        std_floor=args.weather_std_floor,
    )
    print(f"Weather mean (first 3 ch): {weather_mean[:3]}")
    print(f"Weather std  (first 3 ch): {weather_std[:3]}")
    if not tensors_all_finite(weather_mean, weather_std):
        raise RuntimeError("Weather normalization stats contain non-finite values even after skipping bad tensors.")
    if int(weather_mean.numel()) != actual_in_channels or int(weather_std.numel()) != actual_in_channels:
        raise RuntimeError(
            f"Weather-stat channel count mismatch: dataset exposes {actual_in_channels} channels but stats have "
            f"mean={int(weather_mean.numel())}, std={int(weather_std.numel())}."
        )

    runtime_model_config = _build_runtime_model_config(args.seq_len_hist, args.seq_len_fut)
    runtime_model_config["weather_norm_clip"] = None if args.weather_norm_clip <= 0 else float(args.weather_norm_clip)
    runtime_model_config["weather_encoder_force_fp32"] = True
    model_init_cfg = {
        "load_checkpoint": False,
        "return_denormalized": False,
        "in_channels": actual_in_channels,
        "num_zones": train_dataset.num_zones,
        **runtime_model_config,
    }
    model = get_model(model_init_cfg).to(DEVICE)
    if hasattr(model, "set_weather_stats"):
        model.set_weather_stats(weather_mean, weather_std)
    if hasattr(model, "set_target_stats"):
        model.set_target_stats(norm_mean, norm_std, return_denormalized=False)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=amp_enabled)
    ema_model = ModelEma(model, decay=args.ema_decay) if ema_enabled else None

    best_val_mape = float("inf")
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        model.train()
        train_loss_total = 0.0
        train_huber_total = 0.0
        train_mape_loss_total = 0.0
        train_seen = 0
        skipped_train_batches = 0
        optimizer_steps = 0
        accum_counter = 0
        optimizer.zero_grad(set_to_none=True)

        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(train_pbar, start=1):
            weather_seq, hist_demand, hist_cal, fut_cal, target_demand = [b.to(DEVICE, non_blocking=True) for b in batch]
            if not tensors_all_finite(weather_seq, hist_demand, hist_cal, fut_cal, target_demand):
                if args.skip_nonfinite_batches:
                    skipped_train_batches += 1
                    if accum_counter > 0:
                        optimizer.zero_grad(set_to_none=True)
                        accum_counter = 0
                    continue
                raise_nonfinite_batch_error(
                    "train_input",
                    amp_enabled,
                    [("weather_seq", weather_seq), ("hist_demand", hist_demand), ("hist_cal", hist_cal), ("fut_cal", fut_cal), ("target_demand", target_demand)],
                )
            with autocast(device_type=DEVICE.type, enabled=amp_enabled):
                pred_norm = model(weather_seq, hist_demand, hist_cal, fut_cal)
                loss, huber_term, mape_term = compute_training_loss(pred_norm, target_demand, norm_mean.to(DEVICE), norm_std.to(DEVICE))
                loss_to_backward = loss / float(grad_accum_steps)
            if not tensors_all_finite(pred_norm, loss, huber_term, mape_term):
                if args.skip_nonfinite_batches:
                    skipped_train_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0
                    continue
                raise_nonfinite_batch_error(
                    "train_forward",
                    amp_enabled,
                    [("pred_norm", pred_norm), ("loss", loss), ("huber_term", huber_term), ("mape_term", mape_term)],
                )
            scaler.scale(loss_to_backward).backward()
            accum_counter += 1

            should_step = (accum_counter >= grad_accum_steps) or (batch_idx == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if ema_model is not None:
                    ema_model.update(model)
                accum_counter = 0

            batch_size = weather_seq.size(0)
            train_seen += batch_size
            train_loss_total += loss.item() * batch_size
            train_huber_total += float(huber_term.item()) * batch_size
            train_mape_loss_total += float(mape_term.item()) * batch_size
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", skipped=skipped_train_batches, opt_steps=optimizer_steps)

        scheduler.step()
        effective_train_count = max(train_seen, 1)
        avg_train_loss = train_loss_total / effective_train_count
        avg_train_huber = train_huber_total / effective_train_count
        avg_train_mape_term = train_mape_loss_total / effective_train_count

        ema_backup_state = None
        if ema_model is not None:
            ema_backup_state = ema_model.store(model)
            ema_model.copy_to(model)

        model.eval()
        val_loss_total = 0.0
        val_huber_total = 0.0
        val_mape_loss_total = 0.0
        val_mape_total = 0.0
        val_seen = 0
        skipped_val_batches = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validating")
            for batch in val_pbar:
                weather_seq, hist_demand, hist_cal, fut_cal, target_demand = [b.to(DEVICE, non_blocking=True) for b in batch]
                if not tensors_all_finite(weather_seq, hist_demand, hist_cal, fut_cal, target_demand):
                    if args.skip_nonfinite_batches:
                        skipped_val_batches += 1
                        continue
                    raise_nonfinite_batch_error(
                        "val_input",
                        amp_enabled,
                        [("weather_seq", weather_seq), ("hist_demand", hist_demand), ("hist_cal", hist_cal), ("fut_cal", fut_cal), ("target_demand", target_demand)],
                    )
                with autocast(device_type=DEVICE.type, enabled=amp_enabled):
                    pred_norm = model(weather_seq, hist_demand, hist_cal, fut_cal)
                    loss, huber_term, mape_term = compute_training_loss(pred_norm, target_demand, norm_mean.to(DEVICE), norm_std.to(DEVICE))
                pred_raw = unnormalize(pred_norm, norm_mean.to(DEVICE), norm_std.to(DEVICE))
                target_raw = unnormalize(target_demand, norm_mean.to(DEVICE), norm_std.to(DEVICE))
                batch_mape = calculate_mape(pred_raw, target_raw)
                if not tensors_all_finite(pred_norm, pred_raw, target_raw, loss, huber_term, mape_term, batch_mape):
                    if args.skip_nonfinite_batches:
                        skipped_val_batches += 1
                        continue
                    raise_nonfinite_batch_error(
                        "val_forward",
                        amp_enabled,
                        [("pred_norm", pred_norm), ("pred_raw", pred_raw), ("target_raw", target_raw), ("loss", loss), ("huber_term", huber_term), ("mape_term", mape_term), ("batch_mape", batch_mape)],
                    )
                batch_size = weather_seq.size(0)
                val_seen += batch_size
                val_loss_total += loss.item() * batch_size
                val_huber_total += float(huber_term.item()) * batch_size
                val_mape_loss_total += float(mape_term.item()) * batch_size
                val_mape_total += float(batch_mape.item()) * batch_size

        if ema_backup_state is not None:
            ema_model.restore(model, ema_backup_state)

        effective_val_count = max(val_seen, 1)
        avg_val_loss = val_loss_total / effective_val_count
        avg_val_huber = val_huber_total / effective_val_count
        avg_val_mape_term = val_mape_loss_total / effective_val_count
        avg_val_mape = val_mape_total / effective_val_count
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train loss: {avg_train_loss:.4f} | Train huber: {avg_train_huber:.4f} | Train mape-term: {avg_train_mape_term:.4f} | skipped train batches: {skipped_train_batches}"
        )
        print(
            f"Val loss:   {avg_val_loss:.4f} | Val huber:   {avg_val_huber:.4f} | Val mape-term: {avg_val_mape_term:.4f} | Val MAPE: {avg_val_mape:.2f}% | LR: {current_lr:.2e} | skipped val batches: {skipped_val_batches}"
        )

        history_rows.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_huber": avg_train_huber,
            "train_mape_term": avg_train_mape_term,
            "val_loss": avg_val_loss,
            "val_huber": avg_val_huber,
            "val_mape_term": avg_val_mape_term,
            "val_mape": avg_val_mape,
            "lr": current_lr,
            "skipped_train_batches": skipped_train_batches,
            "skipped_val_batches": skipped_val_batches,
            "optimizer_steps": optimizer_steps,
            "grad_accum_steps": grad_accum_steps,
            "ema_enabled": ema_enabled,
            "ema_decay": float(args.ema_decay),
        })
        pd.DataFrame(history_rows).to_csv(output_paths["history_path"], index=False)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": ema_model.state_dict() if ema_model is not None else clone_tensor_state_dict(model.state_dict()),
            "raw_model_state_dict": clone_tensor_state_dict(model.state_dict()),
            "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
            "ema_enabled": ema_enabled,
            "ema_decay": float(args.ema_decay),
            "ema_num_updates": int(ema_model.num_updates) if ema_model is not None else 0,
            "grad_accum_steps": grad_accum_steps,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_mape": avg_val_mape,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "weather_mean": weather_mean,
            "weather_std": weather_std,
            "model_config": {"in_channels": actual_in_channels, "num_zones": train_dataset.num_zones, **runtime_model_config},
            "train_years": train_years,
            "val_years": val_years,
            "manifest_dir": str(manifest_dir),
            "use_manifest": use_manifest,
            "model_name": output_paths['model_name'],
        }
        torch.save(checkpoint, output_paths["latest_checkpoint_path"])
        if avg_val_mape < best_val_mape:
            best_val_mape = avg_val_mape
            torch.save(checkpoint, output_paths["primary_checkpoint_path"])
            torch.save(checkpoint, output_paths["backup_best_path"])
            print(f"[*] New best model saved with Val MAPE: {best_val_mape:.2f}%")
            print(f"    Primary: {output_paths['primary_checkpoint_path']}")
            print(f"    Backup:  {output_paths['backup_best_path']}")


if __name__ == "__main__":
    main()
