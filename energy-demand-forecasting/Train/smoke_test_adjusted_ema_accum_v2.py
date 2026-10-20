from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:
    from generate_dataset_skipbad_debugged import EnergyWeatherDataset, compute_normalization_stats, prebuild_manifests
except ImportError:
    try:
        from generate_dataset_skipbad import EnergyWeatherDataset, compute_normalization_stats, prebuild_manifests
    except ImportError:
        from generate_dataset import EnergyWeatherDataset, compute_normalization_stats  # type: ignore
        prebuild_manifests = None

try:
    from model_adjusted_v3 import get_model
except ImportError:
    try:
        from model_adjusted_v2 import get_model
    except ImportError:
        try:
            from model_adjusted_skipbad_fixed import get_model
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

try:
    from train_adjusted_v3 import (
        MODEL_CONFIG,
        LOSS_CONFIG,
        WEATHER_DIR,
        ENERGY_DIR,
        TRAIN_YEARS,
        VAL_YEARS,
        SELECTED_VARS,
        SEQ_LEN_HIST,
        SEQ_LEN_FUT,
        calculate_mape,
        unnormalize,
        compute_weather_channel_stats,
        compute_training_loss,
        MANIFEST_DIR,
    )
except ImportError:
    try:
        from train_adjusted import (
            MODEL_CONFIG,
            WEATHER_DIR,
            ENERGY_DIR,
            TRAIN_YEARS,
            VAL_YEARS,
            SELECTED_VARS,
            SEQ_LEN_HIST,
            SEQ_LEN_FUT,
            calculate_mape,
            unnormalize,
            compute_weather_channel_stats,
            compute_training_loss,
            MANIFEST_DIR,
        )
        LOSS_CONFIG = {
            "horizon_loss_max_weight": 1.20,
            "horizon_loss_power": 1.0,
            "difficulty_weight_alpha": 0.0,
            "difficulty_weight_gamma": 1.0,
            "difficulty_weight_cap": 3.0,
            "difficulty_weight_mode": "window",
            "ramp_weight_alpha": 0.0,
            "ramp_start_hour": 15,
            "ramp_end_hour": 19,
            "huber_mix": 0.80,
            "mape_mix": 0.20,
        }
    except ImportError:
        WEATHER_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data"
        ENERGY_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data"
        TRAIN_YEARS = [2019, 2020, 2021, 2022]
        VAL_YEARS = [2023]
        SELECTED_VARS: List[str] = []
        SEQ_LEN_HIST = 24
        SEQ_LEN_FUT = 24
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
            "later_horizon_start_step": 8,
            "later_horizon_strength": 0.35,
            "later_horizon_power": 1.5,
            "ramp_start_hour": 15,
            "ramp_end_hour": 19,
            "hour_band_size": 4,
            "use_calendar_second_harmonics": True,
            "zone_adapter_rank": 8,
        }
        LOSS_CONFIG = {
            "horizon_loss_max_weight": 1.20,
            "horizon_loss_power": 1.0,
            "difficulty_weight_alpha": 0.40,
            "difficulty_weight_gamma": 1.0,
            "difficulty_weight_cap": 3.0,
            "difficulty_weight_mode": "window",
            "ramp_weight_alpha": 0.25,
            "ramp_start_hour": 15,
            "ramp_end_hour": 19,
            "huber_mix": 0.80,
            "mape_mix": 0.20,
        }
        MANIFEST_DIR = Path("manifests")

        def calculate_mape(preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
            abs_diff = torch.abs(targets - preds)
            denom = torch.clamp(torch.abs(targets), min=epsilon)
            return (abs_diff / denom).mean() * 100.0

        def unnormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            mean = mean.view(1, 1, -1).to(tensor.device)
            std = std.view(1, 1, -1).to(tensor.device)
            return tensor * std + mean

        def compute_weather_channel_stats(
            weather_dir: Path,
            years: Sequence[int],
            stride: int = 24,
            max_files: Optional[int] = 12,
            manifest_dir: Optional[Path] = None,
            selected_vars: Optional[Sequence[str]] = None,
            rebuild_manifest: bool = False,
            std_floor: float = 0.5,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            files = []
            for year in sorted(set(int(y) for y in years)):
                year_dir = Path(weather_dir) / str(year)
                if year_dir.is_dir():
                    files.extend(sorted(year_dir.glob("X_*.pt")))
            chosen = []
            for i, fpath in enumerate(files):
                if stride > 1 and (i % stride) != 0:
                    continue
                chosen.append(fpath)
                if max_files is not None and len(chosen) >= max_files:
                    break
            sum_channels = None
            sumsq_channels = None
            count = 0
            for fpath in chosen:
                try:
                    x = torch.load(fpath, map_location="cpu", weights_only=True)
                except TypeError:
                    x = torch.load(fpath, map_location="cpu")
                x = x.float()
                if x.ndim != 3 or not torch.isfinite(x).all():
                    continue
                x = x.view(-1, x.shape[-1]).double()
                s = x.sum(dim=0)
                ss = (x * x).sum(dim=0)
                if sum_channels is None:
                    sum_channels = s
                    sumsq_channels = ss
                else:
                    sum_channels += s
                    sumsq_channels += ss
                count += x.shape[0]
            if count == 0 or sum_channels is None or sumsq_channels is None:
                raise RuntimeError("Could not estimate weather stats from finite smoke-test tensors.")
            mean = sum_channels / float(count)
            var = (sumsq_channels / float(count)) - (mean * mean)
            std = torch.sqrt(torch.clamp(var, min=1e-8))
            std = torch.clamp(std, min=float(std_floor))
            return mean.float(), std.float()

        def _cyclical_to_index(cal_feats: torch.Tensor, sin_idx: int, cos_idx: int, period: int) -> torch.Tensor:
            angle = torch.atan2(cal_feats[..., sin_idx], cal_feats[..., cos_idx])
            frac = torch.remainder(angle / (2.0 * torch.pi), 1.0)
            return torch.round(frac * float(period)).long() % period

        def _hours_in_range(hour_idx: torch.Tensor, start_hour: int, end_hour: int) -> torch.Tensor:
            start_hour = int(start_hour) % 24
            end_hour = int(end_hour) % 24
            if start_hour <= end_hour:
                return (hour_idx >= start_hour) & (hour_idx <= end_hour)
            return (hour_idx >= start_hour) | (hour_idx <= end_hour)

        def _build_prev_day_baseline_from_hist(hist_demand: torch.Tensor, fut_len: int) -> torch.Tensor:
            if hist_demand.size(1) >= fut_len:
                return hist_demand[:, -fut_len:, :]
            pad = hist_demand[:, -1:, :].expand(-1, fut_len - hist_demand.size(1), -1)
            return torch.cat([pad, hist_demand], dim=1)[:, -fut_len:, :]

        def compute_training_loss(
            pred_norm: torch.Tensor,
            target_norm: torch.Tensor,
            hist_demand_norm: torch.Tensor,
            fut_cal: torch.Tensor,
            norm_mean: torch.Tensor,
            norm_std: torch.Tensor,
            loss_config: Dict[str, object],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
            fut_len = pred_norm.size(1)
            device = pred_norm.device
            dtype = pred_norm.dtype
            if fut_len <= 1:
                horizon_progress = torch.zeros(fut_len, device=device, dtype=dtype)
            else:
                horizon_progress = torch.linspace(0.0, 1.0, steps=fut_len, device=device, dtype=dtype)
            horizon_weights = 1.0 + (float(loss_config["horizon_loss_max_weight"]) - 1.0) * horizon_progress.pow(
                float(loss_config["horizon_loss_power"])
            )
            point_weights = horizon_weights.view(1, fut_len, 1).expand_as(pred_norm)
            pred_raw = unnormalize(pred_norm, norm_mean, norm_std)
            target_raw = unnormalize(target_norm, norm_mean, norm_std)
            baseline_norm = _build_prev_day_baseline_from_hist(hist_demand_norm, fut_len)
            baseline_raw = unnormalize(baseline_norm, norm_mean, norm_std)
            difficulty_alpha = float(loss_config["difficulty_weight_alpha"])
            difficulty_gamma = float(loss_config["difficulty_weight_gamma"])
            difficulty_cap = float(loss_config["difficulty_weight_cap"])
            difficulty_mode = str(loss_config["difficulty_weight_mode"])
            difficulty_weight_mean = 1.0
            difficulty_extra_max = 0.0
            if difficulty_alpha > 0.0:
                abs_baseline_resid = torch.abs(target_raw - baseline_raw)
                if difficulty_mode == "point":
                    difficulty_source = abs_baseline_resid
                else:
                    difficulty_source = abs_baseline_resid.mean(dim=(1, 2), keepdim=True)
                difficulty_scale = difficulty_source.detach().mean().clamp_min(1e-6)
                relative = difficulty_source / difficulty_scale
                extra = torch.clamp(relative - 1.0, min=0.0, max=difficulty_cap).pow(difficulty_gamma)
                point_weights = point_weights * (1.0 + difficulty_alpha * extra)
                difficulty_weight_mean = float((1.0 + difficulty_alpha * extra).mean().item())
                difficulty_extra_max = float(extra.max().item())
            ramp_alpha = float(loss_config["ramp_weight_alpha"])
            ramp_weight_mean = 1.0
            ramp_fraction = 0.0
            if ramp_alpha > 0.0:
                hour_idx = _cyclical_to_index(fut_cal, 0, 1, 24)
                ramp_mask = _hours_in_range(hour_idx, int(loss_config["ramp_start_hour"]), int(loss_config["ramp_end_hour"])).float().unsqueeze(-1)
                point_weights = point_weights * (1.0 + ramp_alpha * ramp_mask)
                ramp_weight_mean = float((1.0 + ramp_alpha * ramp_mask).mean().item())
                ramp_fraction = float(ramp_mask.mean().item())
            point_weights = point_weights / point_weights.mean().detach().clamp_min(1e-6)
            huber_point = torch.nn.functional.smooth_l1_loss(pred_norm, target_norm, reduction="none")
            weighted_huber = (huber_point * point_weights).mean()
            epsilon = 1e-6
            weighted_mape = (torch.abs(target_raw - pred_raw) / torch.clamp(torch.abs(target_raw), min=epsilon) * point_weights).mean()
            total_loss = float(loss_config["huber_mix"]) * weighted_huber + float(loss_config["mape_mix"]) * weighted_mape
            aux = {
                "avg_point_weight": float(point_weights.mean().item()),
                "max_point_weight": float(point_weights.max().item()),
                "difficulty_weight_mean": difficulty_weight_mean,
                "difficulty_extra_max": difficulty_extra_max,
                "ramp_weight_mean": ramp_weight_mean,
                "ramp_fraction": ramp_fraction,
            }
            return total_loss, weighted_huber.detach(), weighted_mape.detach(), aux


class TinyWindowSubset(Dataset):
    def __init__(self, base_dataset: EnergyWeatherDataset, keep_indices: Sequence[int]):
        self.base_dataset = base_dataset
        self.keep_indices = list(keep_indices)
        self.zone_names = base_dataset.zone_names
        self.num_zones = base_dataset.num_zones
        self.S = base_dataset.S
        self.F = base_dataset.F
        self.norm_mean = base_dataset.norm_mean
        self.norm_std = base_dataset.norm_std

    def __len__(self) -> int:
        return len(self.keep_indices)

    def __getitem__(self, idx: int):
        return self.base_dataset[self.keep_indices[idx]]


class NonFiniteActivationError(RuntimeError):
    def __init__(self, module_name: str, tensor_name: str, stats: dict):
        self.module_name = module_name
        self.tensor_name = tensor_name
        self.stats = stats
        super().__init__(f"Non-finite activation in module '{module_name}' tensor '{tensor_name}'")


def clone_tensor_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, tensor in self._named_state_tensors(model):
            self.shadow[name] = tensor.detach().clone()

    @staticmethod
    def _named_state_tensors(model: torch.nn.Module):
        for name, param in model.named_parameters():
            yield name, param
        for name, buf in model.named_buffers():
            if buf is not None:
                yield name, buf

    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        one_minus_decay = 1.0 - self.decay
        for name, tensor in self._named_state_tensors(model):
            shadow_tensor = self.shadow[name]
            source = tensor.detach()
            if shadow_tensor.is_floating_point():
                shadow_tensor.lerp_(source, one_minus_decay)
            else:
                shadow_tensor.copy_(source)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return clone_tensor_state_dict(self.shadow)


def setup_logger(output_dir: Path) -> Tuple[logging.Logger, Path]:
    log_path = output_dir / "smoke_test_debug.log"
    logger = logging.getLogger(f"smoke_test_debug_{output_dir.resolve()}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_path


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_years(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def tensors_all_finite(*tensors: torch.Tensor) -> bool:
    return all(torch.isfinite(t).all() for t in tensors)


def shrink_dataset_in_place(dataset: EnergyWeatherDataset, max_windows: int, offset: int = 0, step: int = 1) -> List[int]:
    original = list(dataset.valid_start_indices)
    chosen = original[offset::max(step, 1)][:max_windows]
    dataset.valid_start_indices = chosen
    return chosen


def _iter_tensors(obj) -> Iterable[Tuple[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        yield "", obj
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            for sub_name, tensor in _iter_tensors(item):
                name = f"[{i}]" if not sub_name else f"[{i}]{sub_name}"
                yield name, tensor
    elif isinstance(obj, dict):
        for key, item in obj.items():
            for sub_name, tensor in _iter_tensors(item):
                name = f"[{key!s}]" if not sub_name else f"[{key!s}]{sub_name}"
                yield name, tensor


def tensor_debug_stats(tensor: torch.Tensor) -> dict:
    t = tensor.detach()
    stats = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
        "requires_grad": bool(getattr(t, "requires_grad", False)),
    }
    if t.numel() == 0:
        stats.update({
            "is_finite": True,
            "nan_count": 0,
            "posinf_count": 0,
            "neginf_count": 0,
            "finite_min": None,
            "finite_max": None,
            "finite_mean": None,
            "finite_std": None,
            "sample_values": [],
        })
        return stats

    work = t.float() if not t.is_floating_point() else t
    finite_mask = torch.isfinite(work)
    stats["is_finite"] = bool(finite_mask.all().item())
    stats["nan_count"] = int(torch.isnan(work).sum().item())
    stats["posinf_count"] = int(torch.isposinf(work).sum().item())
    stats["neginf_count"] = int(torch.isneginf(work).sum().item())

    finite_vals = work[finite_mask]
    if finite_vals.numel() > 0:
        finite_vals_cpu = finite_vals.detach().cpu()
        stats["finite_min"] = float(finite_vals_cpu.min().item())
        stats["finite_max"] = float(finite_vals_cpu.max().item())
        stats["finite_mean"] = float(finite_vals_cpu.mean().item())
        stats["finite_std"] = float(finite_vals_cpu.std(unbiased=False).item())
        flat_sample = work.detach().reshape(-1)[:8].cpu().tolist()
        stats["sample_values"] = [float(x) for x in flat_sample]
    else:
        stats["finite_min"] = None
        stats["finite_max"] = None
        stats["finite_mean"] = None
        stats["finite_std"] = None
        stats["sample_values"] = []
    return stats


def log_tensor_stats(logger: logging.Logger, name: str, tensor: torch.Tensor, level: int = logging.INFO) -> None:
    logger.log(level, "%s stats: %s", name, json.dumps(tensor_debug_stats(tensor), sort_keys=True))


def log_named_tensor_group(logger: logging.Logger, title: str, named_tensors: Sequence[Tuple[str, torch.Tensor]], level: int = logging.INFO) -> None:
    logger.log(level, "%s", title)
    for name, tensor in named_tensors:
        log_tensor_stats(logger, f"  {name}", tensor, level=level)


def log_model_parameter_diagnostics(logger: logging.Logger, model: torch.nn.Module, level: int = logging.ERROR) -> None:
    bad_params = []
    for name, param in model.named_parameters():
        if not torch.isfinite(param.detach()).all():
            bad_params.append((f"param::{name}", param.detach()))
    for name, buf in model.named_buffers():
        if buf is not None and not torch.isfinite(buf.detach()).all():
            bad_params.append((f"buffer::{name}", buf.detach()))

    if bad_params:
        logger.log(level, "Model contains non-finite parameters or buffers.")
        for name, tensor in bad_params[:20]:
            log_tensor_stats(logger, name, tensor, level=level)
        if len(bad_params) > 20:
            logger.log(level, "Additional non-finite model tensors omitted: %d", len(bad_params) - 20)
    else:
        logger.log(level, "All model parameters and buffers are finite.")


def build_eval_style_raw_inputs(dataset: EnergyWeatherDataset, start_idx: int, history_len: int, future_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if start_idx < history_len:
        raise ValueError(f"Need start_idx >= {history_len} for eval-style smoke test, got {start_idx}")
    hist_slice = slice(start_idx - history_len, start_idx)
    fut_slice = slice(start_idx, start_idx + future_len)
    hist_energy_raw = dataset._targets_raw[hist_slice].clone()
    fut_times = dataset.energy_df.index[fut_slice].values.astype("datetime64[h]").astype("int64")

    hist_weather = []
    fut_weather = []
    hist_keys = dataset.energy_df.index[hist_slice].strftime("%Y%m%d%H").tolist()
    fut_keys = dataset.energy_df.index[fut_slice].strftime("%Y%m%d%H").tolist()
    for key in hist_keys:
        x = dataset._load_weather(dataset.weather_file_lookup[key])
        x = dataset._select_channels(x).float()
        if not torch.isfinite(x).all():
            raise ValueError(f"Non-finite history weather tensor in eval-style smoke input: {key}")
        hist_weather.append(x.contiguous())
    for key in fut_keys:
        x = dataset._load_weather(dataset.weather_file_lookup[key])
        x = dataset._select_channels(x).float()
        if not torch.isfinite(x).all():
            raise ValueError(f"Non-finite future weather tensor in eval-style smoke input: {key}")
        fut_weather.append(x.contiguous())
    return (
        torch.stack(hist_weather, dim=0).unsqueeze(0),
        hist_energy_raw.unsqueeze(0),
        torch.stack(fut_weather, dim=0).unsqueeze(0),
        torch.tensor(fut_times, dtype=torch.int64).unsqueeze(0),
    )


def first_valid_eval_style_start(dataset: EnergyWeatherDataset, min_history_len: int, future_len: int) -> int:
    for start_idx in dataset.valid_start_indices:
        if start_idx < min_history_len or (start_idx + future_len) > len(dataset.energy_df):
            continue
        try:
            build_eval_style_raw_inputs(dataset, int(start_idx), min_history_len, future_len)
            return int(start_idx)
        except Exception:
            continue
    raise RuntimeError("Could not find a valid finite start index for eval-style smoke test.")


def log_window_metadata(logger: logging.Logger, dataset: EnergyWeatherDataset, subset_name: str, keep_indices: Sequence[int]) -> None:
    logger.info("%s subset keep_indices=%s", subset_name, [int(x) for x in keep_indices])
    if not keep_indices:
        return
    for j, start_idx in enumerate(list(keep_indices)[:3]):
        end_idx = int(start_idx) + dataset.S + dataset.F - 1
        start_ts = str(dataset.energy_df.index[int(start_idx)])
        end_ts = str(dataset.energy_df.index[int(end_idx)])
        logger.info("%s window[%d] start_idx=%d end_idx=%d start_ts=%s end_ts=%s", subset_name, j, int(start_idx), end_idx, start_ts, end_ts)


def diagnose_forward_path(
    logger: logging.Logger,
    model: torch.nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    device: torch.device,
    amp_enabled: bool,
) -> None:
    logger.error("Running hooked forward diagnostics with autocast=%s", amp_enabled)
    name_lookup = {module: (name if name else "<root>") for name, module in model.named_modules()}
    handles = []

    def make_hook(module_name: str):
        def _hook(_module, _inputs, output):
            for tensor_name, tensor in _iter_tensors(output):
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise NonFiniteActivationError(module_name, tensor_name or "output", tensor_debug_stats(tensor))
        return _hook

    for name, module in model.named_modules():
        if not name:
            continue
        handles.append(module.register_forward_hook(make_hook(name_lookup[module])))

    try:
        with torch.no_grad():
            with autocast(device_type=device.type, enabled=amp_enabled):
                hooked_pred = model(*inputs)
        logger.error("Hooked forward completed. Final output stats follow.")
        log_tensor_stats(logger, "hooked_pred", hooked_pred, level=logging.ERROR)
    except NonFiniteActivationError as exc:
        logger.error("First non-finite activation detected in module=%s tensor=%s stats=%s", exc.module_name, exc.tensor_name, json.dumps(exc.stats, sort_keys=True))
    except Exception:
        logger.error("Exception while running hooked forward diagnostics:\n%s", traceback.format_exc())
    finally:
        for handle in handles:
            handle.remove()

    if amp_enabled:
        try:
            with torch.no_grad():
                pred_fp32 = model(*inputs)
            if torch.isfinite(pred_fp32).all():
                logger.error("Forward without autocast is finite. This points to mixed precision as the likely trigger.")
            else:
                logger.error("Forward without autocast is still non-finite.")
                log_tensor_stats(logger, "pred_without_autocast", pred_fp32, level=logging.ERROR)
        except Exception:
            logger.error("Exception while running non-autocast diagnostic forward:\n%s", traceback.format_exc())


def raise_with_forward_diagnostics(
    logger: logging.Logger,
    log_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
    weather_seq: torch.Tensor,
    hist_demand: torch.Tensor,
    hist_cal: torch.Tensor,
    fut_cal: torch.Tensor,
    future_demand: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    pred_norm: torch.Tensor,
    loss: torch.Tensor,
    huber_term: torch.Tensor,
    mape_term: torch.Tensor,
    loss_aux: Dict[str, float],
) -> None:
    logger.error("Smoke-test train forward/loss became non-finite.")
    log_named_tensor_group(
        logger,
        "Input batch diagnostics:",
        [
            ("weather_seq", weather_seq),
            ("hist_demand", hist_demand),
            ("hist_cal", hist_cal),
            ("fut_cal", fut_cal),
            ("future_demand", future_demand),
            ("norm_mean", norm_mean),
            ("norm_std", norm_std),
        ],
        level=logging.ERROR,
    )
    log_named_tensor_group(
        logger,
        "Forward/loss diagnostics:",
        [
            ("pred_norm", pred_norm),
            ("loss", loss),
            ("huber_term", huber_term),
            ("mape_term", mape_term),
        ],
        level=logging.ERROR,
    )
    logger.error("loss_aux=%s", json.dumps(loss_aux, sort_keys=True))

    try:
        pred_raw = unnormalize(pred_norm, norm_mean, norm_std)
        target_raw = unnormalize(future_demand, norm_mean, norm_std)
        log_named_tensor_group(
            logger,
            "Denormalized tensors:",
            [("pred_raw", pred_raw), ("target_raw", target_raw)],
            level=logging.ERROR,
        )
    except Exception:
        logger.error("Exception while denormalizing outputs for diagnostics:\n%s", traceback.format_exc())

    log_model_parameter_diagnostics(logger, model, level=logging.ERROR)
    diagnose_forward_path(
        logger=logger,
        model=model,
        inputs=(weather_seq.detach(), hist_demand.detach(), hist_cal.detach(), fut_cal.detach()),
        device=device,
        amp_enabled=amp_enabled,
    )
    raise RuntimeError(f"Smoke-test forward pass produced non-finite train outputs/loss. See debug log: {log_path}")


def build_model_runtime_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "seq_len_hist": int(args.seq_len_hist),
        "seq_len_fut": int(args.seq_len_fut),
        "weather_dim": int(MODEL_CONFIG.get("weather_dim", 128)),
        "tab_dim": int(MODEL_CONFIG.get("tab_dim", 128)),
        "decoder_dim": int(MODEL_CONFIG.get("decoder_dim", 128)),
        "grid_size": tuple(MODEL_CONFIG.get("grid_size", (4, 4))),
        "num_future_layers": int(MODEL_CONFIG.get("num_future_layers", 2)),
        "num_heads": int(MODEL_CONFIG.get("num_heads", 4)),
        "head_hidden": int(MODEL_CONFIG.get("head_hidden", 128)),
        "dropout": float(MODEL_CONFIG.get("dropout", 0.10)),
        "later_horizon_start_step": int(args.later_horizon_start_step),
        "later_horizon_strength": float(args.later_horizon_strength),
        "later_horizon_power": float(args.later_horizon_power),
        "ramp_start_hour": int(args.ramp_start_hour),
        "ramp_end_hour": int(args.ramp_end_hour),
        "hour_band_size": int(args.hour_band_size),
        "use_calendar_second_harmonics": bool(not args.disable_calendar_second_harmonics),
        "zone_adapter_rank": int(args.zone_adapter_rank),
    }


def build_loss_runtime_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "horizon_loss_max_weight": float(args.horizon_loss_max_weight),
        "horizon_loss_power": float(args.horizon_loss_power),
        "difficulty_weight_alpha": float(args.difficulty_weight_alpha),
        "difficulty_weight_gamma": float(args.difficulty_weight_gamma),
        "difficulty_weight_cap": float(args.difficulty_weight_cap),
        "difficulty_weight_mode": str(args.difficulty_weight_mode),
        "ramp_weight_alpha": float(args.ramp_weight_alpha),
        "ramp_start_hour": int(args.ramp_start_hour),
        "ramp_end_hour": int(args.ramp_end_hour),
        "huber_mix": float(args.huber_mix),
        "mape_mix": float(args.mape_mix),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny end-to-end smoke test for the adjusted v3 hybrid model.")
    parser.add_argument("--output-dir", type=str, default="smoke_test_outputs_v3")
    parser.add_argument("--weather-dir", type=str, default=WEATHER_DIR)
    parser.add_argument("--energy-dir", type=str, default=ENERGY_DIR)
    parser.add_argument("--train-years", type=str, default=",".join(map(str, TRAIN_YEARS)))
    parser.add_argument("--val-years", type=str, default=",".join(map(str, VAL_YEARS)))
    parser.add_argument("--max-train-windows", type=int, default=8)
    parser.add_argument("--max-val-windows", type=int, default=4)
    parser.add_argument("--window-step", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weather-stats-stride", type=int, default=48)
    parser.add_argument("--weather-stats-max-files", type=int, default=12)
    parser.add_argument("--eval-history-len", type=int, default=168)
    parser.add_argument("--seq-len-hist", type=int, default=SEQ_LEN_HIST)
    parser.add_argument("--seq-len-fut", type=int, default=SEQ_LEN_FUT)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--weather-std-floor", type=float, default=0.5)
    parser.add_argument("--weather-norm-clip", type=float, default=10.0)
    parser.add_argument("--manifest-dir", type=str, default=str(MANIFEST_DIR))
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--disable-ema", action="store_true")

    parser.add_argument("--later-horizon-start-step", type=int, default=int(MODEL_CONFIG.get("later_horizon_start_step", 8)))
    parser.add_argument("--later-horizon-strength", type=float, default=float(MODEL_CONFIG.get("later_horizon_strength", 0.35)))
    parser.add_argument("--later-horizon-power", type=float, default=float(MODEL_CONFIG.get("later_horizon_power", 1.5)))
    parser.add_argument("--ramp-start-hour", type=int, default=int(MODEL_CONFIG.get("ramp_start_hour", 15)))
    parser.add_argument("--ramp-end-hour", type=int, default=int(MODEL_CONFIG.get("ramp_end_hour", 19)))
    parser.add_argument("--hour-band-size", type=int, default=int(MODEL_CONFIG.get("hour_band_size", 4)))
    parser.add_argument("--disable-calendar-second-harmonics", action="store_true")
    parser.add_argument("--zone-adapter-rank", type=int, default=int(MODEL_CONFIG.get("zone_adapter_rank", 8)))

    parser.add_argument("--horizon-loss-max-weight", type=float, default=float(LOSS_CONFIG.get("horizon_loss_max_weight", 1.20)))
    parser.add_argument("--horizon-loss-power", type=float, default=float(LOSS_CONFIG.get("horizon_loss_power", 1.0)))
    parser.add_argument("--difficulty-weight-alpha", type=float, default=float(LOSS_CONFIG.get("difficulty_weight_alpha", 0.40)))
    parser.add_argument("--difficulty-weight-gamma", type=float, default=float(LOSS_CONFIG.get("difficulty_weight_gamma", 1.0)))
    parser.add_argument("--difficulty-weight-cap", type=float, default=float(LOSS_CONFIG.get("difficulty_weight_cap", 3.0)))
    parser.add_argument("--difficulty-weight-mode", type=str, default=str(LOSS_CONFIG.get("difficulty_weight_mode", "window")), choices=["window", "point"])
    parser.add_argument("--ramp-weight-alpha", type=float, default=float(LOSS_CONFIG.get("ramp_weight_alpha", 0.25)))
    parser.add_argument("--huber-mix", type=float, default=float(LOSS_CONFIG.get("huber_mix", 0.80)))
    parser.add_argument("--mape-mix", type=float, default=float(LOSS_CONFIG.get("mape_mix", 0.20)))

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "artifacts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = setup_logger(output_dir)
    logger.info("Smoke test starting. debug_log=%s", log_path)
    logger.info("Args: %s", json.dumps(vars(args), sort_keys=True))

    device = choose_device(args.device)
    amp_enabled = (device.type == "cuda") and (not args.disable_amp)
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    ema_enabled = not args.disable_ema
    pin_memory = device.type == "cuda"
    train_years = parse_years(args.train_years)
    val_years = parse_years(args.val_years)
    manifest_dir = Path(args.manifest_dir)
    use_manifest = not args.no_manifest
    rebuild_manifest = bool(args.rebuild_manifest)
    model_runtime_config = build_model_runtime_config(args)
    loss_runtime_config = build_loss_runtime_config(args)

    if use_manifest and prebuild_manifests is not None:
        logger.info("Prebuilding manifests for smoke test...")
        prebuild_manifests(
            weather_dir=Path(args.weather_dir),
            energy_dir=Path(args.energy_dir),
            years_groups=[train_years, val_years],
            selected_vars=SELECTED_VARS,
            seq_configs=[(args.seq_len_hist, args.seq_len_fut)],
            manifest_dir=manifest_dir,
            rebuild_manifest=rebuild_manifest,
            verbose=True,
        )

    logger.info("Using device=%s amp_enabled=%s pin_memory=%s", device, amp_enabled, pin_memory)
    logger.info("Building tiny train dataset...")
    train_dataset = EnergyWeatherDataset(
        weather_dir=Path(args.weather_dir),
        energy_dir=Path(args.energy_dir),
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
    logger.info("Train dataset windows before shrink=%d", len(train_dataset))
    train_targets_df = pd.DataFrame(train_dataset._targets_raw.numpy(), columns=train_dataset.zone_names)
    norm_mean, norm_std = compute_normalization_stats(train_targets_df)
    train_dataset.norm_mean = norm_mean
    train_dataset.norm_std = norm_std
    log_named_tensor_group(logger, "Target normalization stats:", [("norm_mean", norm_mean), ("norm_std", norm_std)])

    logger.info("Building tiny val dataset...")
    val_dataset = EnergyWeatherDataset(
        weather_dir=Path(args.weather_dir),
        energy_dir=Path(args.energy_dir),
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
    logger.info("Val dataset windows before shrink=%d", len(val_dataset))

    chosen_train = shrink_dataset_in_place(train_dataset, args.max_train_windows, offset=0, step=args.window_step)
    chosen_val = shrink_dataset_in_place(val_dataset, args.max_val_windows, offset=0, step=args.window_step)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Smoke-test subset ended up empty. Increase max windows or reduce step.")

    log_window_metadata(logger, train_dataset, "train", chosen_train)
    log_window_metadata(logger, val_dataset, "val", chosen_val)

    train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=min(args.batch_size, len(val_dataset)), shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, drop_last=False)

    sample_weather_seq = train_dataset[0][0]
    in_channels = int(sample_weather_seq.shape[1])
    log_tensor_stats(logger, "sample_weather_seq", sample_weather_seq)
    logger.info("Detected in_channels=%d", in_channels)

    weather_mean, weather_std = compute_weather_channel_stats(
        weather_dir=Path(args.weather_dir),
        years=train_years,
        stride=args.weather_stats_stride,
        max_files=args.weather_stats_max_files,
        manifest_dir=manifest_dir,
        selected_vars=SELECTED_VARS,
        rebuild_manifest=rebuild_manifest,
        std_floor=args.weather_std_floor,
    )
    log_named_tensor_group(logger, "Weather normalization stats:", [("weather_mean", weather_mean), ("weather_std", weather_std)])
    if not tensors_all_finite(weather_mean, weather_std):
        raise RuntimeError(f"Smoke-test weather normalization stats are non-finite. See debug log: {log_path}")

    model_cfg = {
        "load_checkpoint": False,
        "return_denormalized": False,
        "in_channels": in_channels,
        "num_zones": train_dataset.num_zones,
        **model_runtime_config,
    }
    logger.info("Model config: %s", json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in model_cfg.items()}, sort_keys=True))
    logger.info("Loss config: %s", json.dumps(loss_runtime_config, sort_keys=True))
    model = get_model(model_cfg).to(device)
    if hasattr(model, "set_weather_stats"):
        model.set_weather_stats(weather_mean, weather_std)
    if hasattr(model, "set_target_stats"):
        model.set_target_stats(norm_mean, norm_std, return_denormalized=False)
    log_model_parameter_diagnostics(logger, model, level=logging.INFO)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler(enabled=amp_enabled)
    ema_model = ModelEma(model, decay=args.ema_decay) if ema_enabled else None

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_batch = None
    pred_norm = None
    loss = None
    huber_term = None
    mape_term = None
    loss_aux: Dict[str, float] = {}
    optimizer_steps = 0
    accum_batches_used = 0
    grad_norm = 0.0

    for train_batch in train_loader:
        logger.info("Fetched train batch %d for smoke-step accumulation.", accum_batches_used + 1)
        weather_seq, hist_demand, hist_cal, fut_cal, future_demand = [b.to(device, non_blocking=True) for b in train_batch]
        if accum_batches_used == 0:
            log_named_tensor_group(
                logger,
                "First train batch tensors:",
                [
                    ("weather_seq", weather_seq),
                    ("hist_demand", hist_demand),
                    ("hist_cal", hist_cal),
                    ("fut_cal", fut_cal),
                    ("future_demand", future_demand),
                ],
            )
        if not tensors_all_finite(weather_seq, hist_demand, hist_cal, fut_cal, future_demand):
            raise RuntimeError(f"Smoke-test train batch contains non-finite values after dataset filtering. See debug log: {log_path}")

        with autocast(device_type=device.type, enabled=amp_enabled):
            pred_norm = model(weather_seq, hist_demand, hist_cal, fut_cal)
            loss, huber_term, mape_term, loss_aux = compute_training_loss(
                pred_norm,
                future_demand,
                hist_demand,
                fut_cal,
                norm_mean.to(device),
                norm_std.to(device),
                loss_runtime_config,
            )
            loss_to_backward = loss / float(grad_accum_steps)

        if not tensors_all_finite(pred_norm, loss, huber_term, mape_term):
            raise_with_forward_diagnostics(
                logger=logger,
                log_path=log_path,
                model=model,
                device=device,
                amp_enabled=amp_enabled,
                weather_seq=weather_seq,
                hist_demand=hist_demand,
                hist_cal=hist_cal,
                fut_cal=fut_cal,
                future_demand=future_demand,
                norm_mean=norm_mean.to(device),
                norm_std=norm_std.to(device),
                pred_norm=pred_norm,
                loss=loss,
                huber_term=huber_term,
                mape_term=mape_term,
                loss_aux=loss_aux,
            )

        scaler.scale(loss_to_backward).backward()
        accum_batches_used += 1
        should_step = (accum_batches_used >= grad_accum_steps) or (accum_batches_used == len(train_loader))
        if should_step:
            scaler.unscale_(optimizer)
            grad_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_sq += float(torch.sum(p.grad.detach().float() ** 2).item())
            grad_norm = grad_norm_sq ** 0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if ema_model is not None:
                ema_model.update(model)
            logger.info(
                "Completed one optimizer step with grad_accum_steps=%d after %d batch(es). grad_l2_norm=%.6f loss_aux=%s",
                grad_accum_steps,
                accum_batches_used,
                grad_norm,
                json.dumps(loss_aux, sort_keys=True),
            )
            break

    if train_batch is None or pred_norm is None or loss is None or huber_term is None or mape_term is None:
        raise RuntimeError(f"Smoke-test did not process any training batches. See debug log: {log_path}")

    with torch.no_grad():
        pred_raw = unnormalize(pred_norm, norm_mean.to(device), norm_std.to(device))
        target_raw = unnormalize(future_demand, norm_mean.to(device), norm_std.to(device))
        train_batch_mape = float(calculate_mape(pred_raw, target_raw).item())
        log_named_tensor_group(logger, "Post-step raw tensors:", [("pred_raw", pred_raw), ("target_raw", target_raw)])
        logger.info("Train batch MAPE=%.6f", train_batch_mape)

    model.eval()
    val_losses = []
    val_mapes = []
    val_aux_records = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            weather_seq, hist_demand, hist_cal, fut_cal, future_demand = [b.to(device, non_blocking=True) for b in batch]
            with autocast(device_type=device.type, enabled=amp_enabled):
                pred_norm = model(weather_seq, hist_demand, hist_cal, fut_cal)
                vloss, _, _, vaux = compute_training_loss(
                    pred_norm,
                    future_demand,
                    hist_demand,
                    fut_cal,
                    norm_mean.to(device),
                    norm_std.to(device),
                    loss_runtime_config,
                )
            pred_raw = unnormalize(pred_norm, norm_mean.to(device), norm_std.to(device))
            target_raw = unnormalize(future_demand, norm_mean.to(device), norm_std.to(device))
            batch_mape = calculate_mape(pred_raw, target_raw)
            if not tensors_all_finite(pred_norm, pred_raw, target_raw, vloss, batch_mape):
                logger.error("Validation batch %d produced non-finite tensors.", batch_idx)
                log_named_tensor_group(
                    logger,
                    "Validation batch diagnostics:",
                    [
                        ("weather_seq", weather_seq),
                        ("hist_demand", hist_demand),
                        ("hist_cal", hist_cal),
                        ("fut_cal", fut_cal),
                        ("future_demand", future_demand),
                        ("pred_norm", pred_norm),
                        ("pred_raw", pred_raw),
                        ("target_raw", target_raw),
                        ("vloss", vloss),
                        ("batch_mape", batch_mape),
                    ],
                    level=logging.ERROR,
                )
                raise RuntimeError(f"Smoke-test validation produced non-finite outputs/loss. See debug log: {log_path}")
            val_losses.append(float(vloss.item()))
            val_mapes.append(float(batch_mape.item()))
            val_aux_records.append({k: float(v) for k, v in vaux.items()})

    ckpt_path = ckpt_dir / "smoke_best_model_v3.pth"
    checkpoint = {
        "epoch": 0,
        "model_state_dict": ema_model.state_dict() if ema_model is not None else clone_tensor_state_dict(model.state_dict()),
        "raw_model_state_dict": clone_tensor_state_dict(model.state_dict()),
        "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
        "ema_enabled": ema_enabled,
        "ema_decay": float(args.ema_decay),
        "ema_num_updates": int(ema_model.num_updates) if ema_model is not None else 0,
        "grad_accum_steps": grad_accum_steps,
        "optimizer_state_dict": optimizer.state_dict(),
        "val_mape": float(sum(val_mapes) / max(len(val_mapes), 1)),
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "weather_mean": weather_mean,
        "weather_std": weather_std,
        "model_config": {"in_channels": in_channels, "num_zones": train_dataset.num_zones, **model_runtime_config},
        "loss_config": dict(loss_runtime_config),
        "train_years": train_years,
        "val_years": val_years,
        "manifest_dir": str(manifest_dir),
        "use_manifest": use_manifest,
        "smoke_test_args": vars(args),
    }
    torch.save(checkpoint, ckpt_path)
    logger.info("Checkpoint saved to %s", ckpt_path)

    reloaded_model = get_model({"checkpoint_path": ckpt_path, "load_checkpoint": True, "return_denormalized": False}).to(device)
    reloaded_model.eval()
    with torch.no_grad():
        weather_seq, hist_demand, hist_cal, fut_cal, _ = [b.to(device, non_blocking=True) for b in train_batch]
        reloaded_pred = reloaded_model(weather_seq, hist_demand, hist_cal, fut_cal)
        if not tensors_all_finite(reloaded_pred):
            log_tensor_stats(logger, "reloaded_pred", reloaded_pred, level=logging.ERROR)
            raise RuntimeError(f"Reloaded smoke-test model produced non-finite outputs. See debug log: {log_path}")

    eval_start_idx = first_valid_eval_style_start(train_dataset, args.eval_history_len, args.seq_len_fut)
    logger.info("Using eval-style start_idx=%d", eval_start_idx)
    raw_hist_weather, raw_hist_energy, raw_fut_weather, raw_fut_time = build_eval_style_raw_inputs(train_dataset, eval_start_idx, args.eval_history_len, args.seq_len_fut)
    raw_hist_weather = raw_hist_weather.to(device)
    raw_hist_energy = raw_hist_energy.to(device)
    raw_fut_weather = raw_fut_weather.to(device)
    raw_fut_time = raw_fut_time.to(device)

    with torch.no_grad():
        adapted = reloaded_model.adapt_inputs(raw_hist_weather, raw_hist_energy, raw_fut_weather, raw_fut_time)
        adapt_shapes = [tuple(x.shape) for x in adapted]
        for i, x in enumerate(adapted):
            if isinstance(x, torch.Tensor):
                log_tensor_stats(logger, f"adapted[{i}]", x)
        if not all(torch.isfinite(x).all() for x in adapted if isinstance(x, torch.Tensor)):
            raise RuntimeError(f"Adapted eval-style smoke inputs contain non-finite values. See debug log: {log_path}")
        eval_style_pred = reloaded_model(*adapted)
        if not tensors_all_finite(eval_style_pred):
            log_tensor_stats(logger, "eval_style_pred", eval_style_pred, level=logging.ERROR)
            raise RuntimeError(f"Eval-style smoke prediction is non-finite. See debug log: {log_path}")

    summary = {
        "success": True,
        "device": str(device),
        "debug_log": str(log_path),
        "train_years": train_years,
        "val_years": val_years,
        "manifest_dir": str(manifest_dir),
        "use_manifest": use_manifest,
        "rebuild_manifest": rebuild_manifest,
        "train_windows_used": len(train_dataset),
        "val_windows_used": len(val_dataset),
        "train_subset_start_indices": [int(x) for x in chosen_train],
        "val_subset_start_indices": [int(x) for x in chosen_val],
        "in_channels": in_channels,
        "num_zones": int(train_dataset.num_zones),
        "model_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in model_runtime_config.items()},
        "loss_config": dict(loss_runtime_config),
        "train_batch_shapes": {
            "weather_seq": list(train_batch[0].shape),
            "hist_demand": list(train_batch[1].shape),
            "hist_cal": list(train_batch[2].shape),
            "fut_cal": list(train_batch[3].shape),
            "future_demand": list(train_batch[4].shape),
        },
        "single_step": {
            "loss": float(loss.item()),
            "huber_term": float(huber_term.item()),
            "mape_term": float(mape_term.item()),
            "loss_aux": {k: float(v) for k, v in loss_aux.items()},
            "train_batch_mape": train_batch_mape,
            "gradient_l2_norm": grad_norm,
            "optimizer_steps": optimizer_steps,
            "grad_accum_steps": grad_accum_steps,
            "accum_batches_used": accum_batches_used,
        },
        "validation": {
            "num_batches": len(val_losses),
            "mean_loss": float(sum(val_losses) / max(len(val_losses), 1)),
            "mean_mape": float(sum(val_mapes) / max(len(val_mapes), 1)),
            "mean_loss_aux": {
                key: float(sum(rec.get(key, 0.0) for rec in val_aux_records) / max(len(val_aux_records), 1))
                for key in ([*val_aux_records[0].keys()] if val_aux_records else [])
            },
        },
        "checkpoint_roundtrip": {"path": str(ckpt_path), "reloaded_forward_shape": list(reloaded_pred.shape)},
        "ema": {
            "enabled": ema_enabled,
            "decay": float(args.ema_decay),
            "num_updates": int(ema_model.num_updates) if ema_model is not None else 0,
        },
        "evaluate_compatibility": {
            "history_len_passed": int(args.eval_history_len),
            "adapted_shapes": [list(s) for s in adapt_shapes],
            "prediction_shape": list(eval_style_pred.shape),
        },
    }

    summary_path = output_dir / "smoke_test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Smoke test passed. Summary written to %s", summary_path)
    print("\nSmoke test passed.")
    print(f"Train windows used: {len(train_dataset)} | Val windows used: {len(val_dataset)}")
    print(f"Single-step loss: {summary['single_step']['loss']:.4f} | train-batch MAPE: {summary['single_step']['train_batch_mape']:.2f}%")
    print(f"Validation mean loss: {summary['validation']['mean_loss']:.4f} | mean MAPE: {summary['validation']['mean_mape']:.2f}%")
    print(f"Reloaded forward shape: {tuple(reloaded_pred.shape)}")
    print(f"Eval-style prediction shape: {tuple(eval_style_pred.shape)}")
    print(f"Summary written to: {summary_path}")
    print(f"Debug log written to: {log_path}")


if __name__ == "__main__":
    main()
