#!/usr/bin/env python3
"""
extract_extreme_errors.py
=========================
Run inference on a validation year and save two CSVs:

  1. predictions.csv
     One row per (window, forecast_hour, zone).
     Columns: window_start_utc, forecast_utc, forecast_hour, zone,
              true_mwh, pred_mwh, signed_error_mwh, abs_error_mwh, pct_error

  2. weather_stats.csv
     One row per window.
     Columns: window_start_utc,
              ch{i}_fut_mean   <- raw spatial mean of channel i, averaged over F future steps
              ch{i}_fut_max    <- raw spatial max,  averaged over F future steps
     These are passed to analyze_extreme_errors.py, which applies percentile
     thresholds to decide what counts as "extreme".

Usage (on the cluster):
    python extract_extreme_errors.py \
        --checkpoint checkpoints/model_1_v2_ema_accum_best.pth \
        --model-file  model_adjusted_v2.py \
        --years 2023 \
        --seq-len-hist 24 \
        --output-dir  ./extreme_analysis \
        --batch-size  8 \
        --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ── Cluster paths (override with CLI flags if needed) ────────────────────────
WEATHER_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data"
ENERGY_DIR  = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_model_module(model_file: str):
    """Dynamically load get_model from an arbitrary .py path."""
    spec = importlib.util.spec_from_file_location("_model_mod", model_file)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unnormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Reverse z-score normalization.  Tensors are broadcast-safe."""
    return x * std.to(x.device) + mean.to(x.device)


class _IndexedDataset(Dataset):
    """Wraps EnergyWeatherDataset to also return the dataset-level index."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return (idx, *self.ds[idx])


def _try_load_channel_names(weather_dir: Path) -> Optional[List[str]]:
    meta_path = Path(weather_dir).parent / "metadata.pt"
    if not meta_path.exists():
        return None
    try:
        meta  = torch.load(meta_path, map_location="cpu", weights_only=False)
        names = [str(n) for n in meta.get("variable_names", [])]
        return names if names else None
    except Exception:
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Extract extreme-weather error samples.")
    p.add_argument("--checkpoint",    required=True,  help="Path to .pth checkpoint file.")
    p.add_argument("--model-file",    required=True,  help="Path to model_adjusted_vX.py.")
    p.add_argument("--weather-dir",   default=WEATHER_DIR)
    p.add_argument("--energy-dir",    default=ENERGY_DIR)
    p.add_argument("--manifest-dir",  default="",     help="Leave empty to auto-detect.")
    p.add_argument("--years",         default="2023", help="Comma-separated validation years.")
    p.add_argument("--seq-len-hist",  type=int, default=24)
    p.add_argument("--batch-size",    type=int, default=8)
    p.add_argument("--num-workers",   type=int, default=0)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir",    default="./extreme_analysis")
    return p.parse_args()


def main():
    args    = parse_args()
    device  = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    print(f"[*] Validation years : {years}")
    print(f"[*] Device           : {device}")
    print(f"[*] Checkpoint       : {args.checkpoint}")

    # ── 1. Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    norm_mean    = ckpt["norm_mean"].float()          # (Z,)
    norm_std     = ckpt["norm_std"].float()           # (Z,)
    weather_mean = ckpt["weather_mean"].float()       # (C,)
    weather_std  = ckpt["weather_std"].float()        # (C,)
    saved_config = dict(ckpt.get("model_config", {}))
    print(f"[*] norm_mean shape  : {norm_mean.shape}")
    print(f"[*] weather_mean shape: {weather_mean.shape}")

    num_channels = int(weather_mean.numel())

    # ── 2. Load dataset ──────────────────────────────────────────────────────
    # Import EnergyWeatherDataset from generate_dataset.py (same folder as model)
    model_dir = Path(args.model_file).resolve().parent
    sys.path.insert(0, str(model_dir))
    from generate_dataset import EnergyWeatherDataset  # noqa: E402

    manifest_dir_arg = Path(args.manifest_dir) if args.manifest_dir else None

    print("[*] Building validation dataset …")
    val_dataset = EnergyWeatherDataset(
        weather_dir    = Path(args.weather_dir),
        energy_dir     = Path(args.energy_dir),
        years          = years,
        seq_len_hist   = args.seq_len_hist,
        seq_len_fut    = 24,
        manifest_dir   = manifest_dir_arg,
        use_manifest   = True,
        rebuild_manifest = False,
        verbose        = True,
    )
    # Assign demand normalization stats so hist_demand and target_demand
    # come out already z-scored (matching what the model expects).
    val_dataset.norm_mean = norm_mean
    val_dataset.norm_std  = norm_std

    zone_names = val_dataset.zone_names
    S          = val_dataset.S        # historical window length
    F          = val_dataset.F        # forecast horizon (24)
    print(f"[*] Zones            : {zone_names}")
    print(f"[*] S={S}, F={F}, windows={len(val_dataset)}")

    channel_names = _try_load_channel_names(Path(args.weather_dir))
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(num_channels)]
    print(f"[*] Weather channels : {channel_names}")

    # ── 3. Build model ───────────────────────────────────────────────────────
    model_mod = load_model_module(args.model_file)
    metadata  = {
        "load_checkpoint"      : True,
        "checkpoint_path"      : args.checkpoint,
        "return_denormalized"  : False,   # we unnormalize manually
        **saved_config,
    }
    model = model_mod.get_model(metadata)
    model.set_weather_stats(weather_mean, weather_std)
    model.set_target_stats(norm_mean, norm_std, return_denormalized=False)
    model.to(device)
    model.eval()
    print("[*] Model loaded successfully.")

    # ── 4. Inference loop ────────────────────────────────────────────────────
    indexed_ds = _IndexedDataset(val_dataset)
    loader = DataLoader(
        indexed_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
        drop_last   = False,
    )

    all_preds   = []   # list of dicts -> predictions.csv
    all_wx_stats = []  # list of dicts -> weather_stats.csv

    # Pre-compute (B, C) broadcast tensors for unnormalize
    nm = norm_mean.view(1, 1, -1)   # (1, 1, Z)
    ns = norm_std.view(1, 1, -1)    # (1, 1, Z)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            (
                batch_idx,        # (B,)  dataset-level indices
                weather_seq,      # (B, S+F, C, H, W)  RAW (not normalised)
                hist_demand,      # (B, S, Z)  normalised
                hist_cal,         # (B, S, 4)
                fut_cal,          # (B, F, 4)
                target_demand,    # (B, F, Z)  normalised
            ) = batch

            # Move to device
            weather_seq   = weather_seq.to(device)
            hist_demand   = hist_demand.to(device)
            hist_cal      = hist_cal.to(device)
            fut_cal       = fut_cal.to(device)
            target_demand = target_demand.to(device)

            # Forward pass → normalised predictions (B, F, Z)
            pred_norm = model(weather_seq, hist_demand, hist_cal, fut_cal)

            # Unnormalise both predictions and targets to raw MWh
            pred_raw   = unnormalize(pred_norm,   nm, ns)   # (B, F, Z)
            target_raw = unnormalize(target_demand, nm, ns)  # (B, F, Z)

            # Move back to CPU for numpy/pandas work
            pred_raw   = pred_raw.cpu().float()
            target_raw = target_raw.cpu().float()
            batch_idx  = batch_idx.cpu()

            # ── Weather spatial statistics for future frames ─────────────────
            # weather_seq is (B, S+F, C, H, W), grab future portion in RAW space
            # (model normalises internally; we want raw for interpretable stats)
            fut_wx = weather_seq[:, S:, :, :, :].cpu().float()   # (B, F, C, H, W)
            # Spatial mean per channel per step  →  (B, F, C)
            fut_wx_spatial_mean = fut_wx.mean(dim=(-2, -1))       # (B, F, C)
            # Spatial max per channel per step   →  (B, F, C)
            fut_wx_spatial_max  = fut_wx.amax(dim=(-2, -1))       # (B, F, C)
            # Average across forecast horizon    →  (B, C)
            wx_mean_over_f = fut_wx_spatial_mean.mean(dim=1)      # (B, C)
            wx_max_over_f  = fut_wx_spatial_max.mean(dim=1)       # (B, C)

            B = weather_seq.size(0)
            for b in range(B):
                ds_idx        = int(batch_idx[b])
                valid_start   = val_dataset.valid_start_indices[ds_idx]
                window_start_ts = val_dataset.energy_df.index[valid_start]

                # ── Weather stats row ─────────────────────────────────────
                wx_row = {"window_start_utc": window_start_ts}
                for ci, cname in enumerate(channel_names):
                    safe = cname.replace(" ", "_")
                    wx_row[f"{safe}_fut_mean"] = float(wx_mean_over_f[b, ci])
                    wx_row[f"{safe}_fut_max"]  = float(wx_max_over_f[b, ci])
                all_wx_stats.append(wx_row)

                # ── Prediction rows ───────────────────────────────────────
                for fi in range(F):
                    # Timestamp of this forecast step
                    forecast_ts = val_dataset.energy_df.index[valid_start + S + fi]
                    for zi, zname in enumerate(zone_names):
                        true_v   = float(target_raw[b, fi, zi])
                        pred_v   = float(pred_raw[b, fi, zi])
                        signed_e = pred_v - true_v
                        abs_e    = abs(signed_e)
                        pct_e    = abs_e / max(abs(true_v), 1e-6) * 100.0
                        all_preds.append({
                            "window_start_utc"  : window_start_ts,
                            "forecast_utc"      : forecast_ts,
                            "forecast_hour"     : fi + 1,          # 1-indexed
                            "zone"              : zname,
                            "true_mwh"          : true_v,
                            "pred_mwh"          : pred_v,
                            "signed_error_mwh"  : signed_e,
                            "abs_error_mwh"     : abs_e,
                            "pct_error"         : pct_e,
                        })

    # ── 5. Save CSVs ─────────────────────────────────────────────────────────
    pred_df   = pd.DataFrame(all_preds)
    wx_df     = pd.DataFrame(all_wx_stats)

    pred_csv  = out_dir / "predictions.csv"
    wx_csv    = out_dir / "weather_stats.csv"
    meta_path = out_dir / "extraction_meta.pt"

    pred_df.to_csv(pred_csv,  index=False)
    wx_df.to_csv(wx_csv,      index=False)

    # Save channel names so the analysis script can read them without metadata.pt
    torch.save({
        "channel_names" : channel_names,
        "zone_names"    : zone_names,
        "years"         : years,
        "S"             : S,
        "F"             : F,
    }, meta_path)

    print(f"\n[✓] Saved {len(pred_df):,} prediction rows  →  {pred_csv}")
    print(f"[✓] Saved {len(wx_df):,} weather-stat rows →  {wx_csv}")
    print(f"[✓] Metadata                              →  {meta_path}")
    print("\nNext step:\n  python analyze_extreme_errors.py "
          f"--output-dir {out_dir}")


if __name__ == "__main__":
    main()