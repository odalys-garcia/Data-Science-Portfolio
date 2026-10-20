
#!/usr/bin/env python3
"""
Extended dataset + inference-only analysis script for the energy demand forecasting project.

What this adds on top of the previous dataset analysis script
-------------------------------------------------------------
1) Keeps the original dataset profiling:
   - contiguity / split coverage
   - valid window counts
   - per-zone stats
   - regime effects
   - cross-zone dependence
   - autocorrelation / seasonality
   - ramp behavior
   - naive baseline difficulty
   - sampled weather-channel summaries

2) Adds inference-only analysis for an already-trained checkpoint:
   - full model evaluation on all available windows for chosen years
   - branch ablations with NO retraining
   - effective-history masking ablations within the model's used history span
   - optional per-channel weather knockout ablations
   - current-model vs prev-day baseline comparison
   - error slicing by zone / horizon / local hour / weekend / load quantile / ramp quantile
   - cheap post-hoc blend analysis:
       * model
       * prev_day baseline
       * global blend fitted on calibration windows
       * per-zone blend fitted on calibration windows
       * per-zone-per-horizon blend fitted on calibration windows

This is designed around the current project structure used by:
- generate_dataset.py
- train.py
- evaluate.py
- model.py

No full retraining is done.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pandas.tseries.holiday import USFederalHolidayCalendar
from tqdm.auto import tqdm


# ============================================================
# Default paths aligned with the current project structure
# ============================================================
WEATHER_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data"
ENERGY_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data"

TRAIN_YEARS = [2019, 2020, 2021, 2022]
VAL_YEARS = [2023]
TEST_YEARS = [2024]


# ============================================================
# Optional reuse of existing project helper if available
# ============================================================
try:
    from generate_dataset import load_energy_targets as _project_load_energy_targets  # type: ignore
except Exception:
    _project_load_energy_targets = None


# ============================================================
# Generic helpers
# ============================================================
def parse_years(years_text: str) -> List[int]:
    years_text = years_text.strip()
    if not years_text:
        return []
    return [int(x.strip()) for x in years_text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def safe_torch_load(path: Path) -> torch.Tensor:
    try:
        x = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        x = torch.load(path, map_location="cpu")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, found {type(x)}")
    return x.float()


def mape_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(target), eps)
    return float(np.mean(np.abs(pred - target) / denom) * 100.0)


def smape_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(pred) + np.abs(target), eps)
    return float(np.mean(2.0 * np.abs(pred - target) / denom) * 100.0)


def mae_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def rmse_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def bias_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(pred - target))


def load_energy_targets(energy_dir: Path, years: Sequence[int]) -> pd.DataFrame:
    """Load energy targets using the project's helper when available."""
    if _project_load_energy_targets is not None:
        df = _project_load_energy_targets(energy_dir, list(years))
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Loaded energy dataframe must be indexed by datetime.")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df.sort_index()

    frames = []
    for year in years:
        fpath = energy_dir / f"target_energy_zonal_{year}.csv"
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        dt_col = df.columns[0]
        for col in df.columns:
            if col.strip().lower() in ("datetime", "timestamp", "time", "date", "timestamp_utc"):
                dt_col = col
                break
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
        df = df.set_index(dt_col)
        df.index.name = "datetime"
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No energy CSVs found in {energy_dir} for years {years}")

    return pd.concat(frames).sort_index()


def maybe_load_metadata(weather_root: Path) -> List[str]:
    meta_path = weather_root.parent / "metadata.pt"
    if not meta_path.exists():
        return []
    try:
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        names = list(meta.get("variable_names", []))
        return [str(x) for x in names]
    except Exception:
        return []


def build_weather_lookup(weather_dir: Path, years: Sequence[int]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for year in sorted(set(int(y) for y in years)):
        year_dir = weather_dir / str(year)
        if not year_dir.is_dir():
            continue
        for fpath in year_dir.glob("X_*.pt"):
            lookup[fpath.stem[2:]] = fpath
    return lookup


def infer_num_weather_vars(weather_lookup: Dict[str, Path]) -> int:
    if not weather_lookup:
        raise RuntimeError("No weather files found.")
    first_path = weather_lookup[sorted(weather_lookup.keys())[0]]
    x = safe_torch_load(first_path)
    if x.ndim != 3:
        raise ValueError(f"Expected weather tensor with 3 dims, got {tuple(x.shape)}")
    return int(x.shape[-1])


def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    local_idx = idx.tz_convert("US/Eastern")

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=idx.min().date(), end=idx.max().date())
    holiday_dates = set(pd.DatetimeIndex(holidays).date)

    out["timestamp_utc"] = idx
    out["timestamp_local"] = local_idx
    out["hour_utc"] = idx.hour
    out["hour_local"] = local_idx.hour
    out["dow_utc"] = idx.dayofweek
    out["dow_local"] = local_idx.dayofweek
    out["month"] = local_idx.month
    out["day_of_year"] = local_idx.dayofyear
    out["year_local"] = local_idx.year
    out["is_weekend"] = (local_idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = np.array([int(ts.date() in holiday_dates) for ts in local_idx], dtype=np.int64)
    return out


def compute_contiguity_info(idx: pd.DatetimeIndex) -> Dict[str, object]:
    deltas = idx.to_series().diff().dropna()
    one_hour = pd.Timedelta(hours=1)
    gap_mask = deltas != one_hour
    info = {
        "is_contiguous_hourly": bool((~gap_mask).all()),
        "num_rows": int(len(idx)),
        "start": str(idx.min()),
        "end": str(idx.max()),
        "num_gaps": int(gap_mask.sum()),
    }
    if gap_mask.any():
        gaps = deltas[gap_mask]
        first_gaps = []
        for ts, delta in gaps.head(10).items():
            first_gaps.append({"timestamp": str(ts), "delta": str(delta)})
        info["first_gaps"] = first_gaps
    return info


# ============================================================
# Dataset profiling
# ============================================================
def compute_window_validity(
    energy_df: pd.DataFrame,
    weather_lookup: Dict[str, Path],
    hist_len: int,
    fut_len: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_len = hist_len + fut_len
    keys = energy_df.index.strftime("%Y%m%d%H")
    target_values = energy_df.to_numpy(dtype=np.float32)
    target_nan = np.isnan(target_values).any(axis=1)
    weather_exists = np.array([k in weather_lookup for k in keys], dtype=bool)
    step_valid = (~target_nan) & weather_exists

    timestamps = energy_df.index
    start_indices: List[int] = []
    start_times: List[pd.Timestamp] = []
    for i in range(len(energy_df) - total_len + 1):
        j = i + total_len - 1
        if not step_valid[i : i + total_len].all():
            continue
        if (timestamps[j] - timestamps[i]) != pd.Timedelta(hours=total_len - 1):
            continue
        start_indices.append(i)
        start_times.append(timestamps[i])

    detail_df = pd.DataFrame({
        "window_start": start_times,
        "window_end": [timestamps[i + total_len - 1] for i in start_indices],
    })
    if not detail_df.empty:
        detail_df["start_year_utc"] = detail_df["window_start"].dt.year
        detail_df["start_month_utc"] = detail_df["window_start"].dt.month

    summary = []
    year_groups = energy_df.index.year
    for year in sorted(pd.unique(year_groups)):
        year_mask = year_groups == year
        summary.append({
            "year": int(year),
            "num_timesteps": int(year_mask.sum()),
            "num_target_nan_steps": int(target_nan[year_mask].sum()),
            "num_weather_missing_steps": int((~weather_exists[year_mask]).sum()),
            "num_valid_steps": int(step_valid[year_mask].sum()),
            "num_valid_windows_starting_in_year": int((detail_df["start_year_utc"] == year).sum()) if not detail_df.empty else 0,
        })
    return pd.DataFrame(summary), detail_df


def compute_zone_stats(df: pd.DataFrame, zone_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for zone in zone_cols:
        s = pd.to_numeric(df[zone], errors="coerce")
        rows.append({
            "zone": zone,
            "count": int(s.notna().sum()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "cv": float(s.std() / max(abs(float(s.mean())), 1e-6)),
            "missing_frac": float(s.isna().mean()),
        })
    return pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)


def compute_group_profiles(df: pd.DataFrame, zone_cols: Sequence[str], group_col: str) -> pd.DataFrame:
    g = df.groupby(group_col)[list(zone_cols)]
    mean_df = g.mean().reset_index()
    std_df = g.std().reset_index()
    long_mean = mean_df.melt(id_vars=[group_col], var_name="zone", value_name="mean_load")
    long_std = std_df.melt(id_vars=[group_col], var_name="zone", value_name="std_load")
    out = long_mean.merge(long_std, on=[group_col, "zone"], how="left")
    return out.sort_values([group_col, "zone"]).reset_index(drop=True)


def compute_regime_effect_summary(df: pd.DataFrame, zone_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for zone in zone_cols:
        s = df[zone]
        weekday_mean = float(s[df["is_weekend"] == 0].mean())
        weekend_mean = float(s[df["is_weekend"] == 1].mean())
        holiday_mean = float(s[df["is_holiday"] == 1].mean()) if (df["is_holiday"] == 1).any() else math.nan
        nonholiday_mean = float(s[df["is_holiday"] == 0].mean())

        by_hour = df.groupby("hour_local")[zone].mean()
        by_month = df.groupby("month")[zone].mean()

        rows.append({
            "zone": zone,
            "weekday_mean": weekday_mean,
            "weekend_mean": weekend_mean,
            "weekend_vs_weekday_pct": float(100.0 * (weekend_mean - weekday_mean) / max(abs(weekday_mean), 1e-6)),
            "holiday_mean": holiday_mean,
            "nonholiday_mean": nonholiday_mean,
            "holiday_vs_nonholiday_pct": float(100.0 * (holiday_mean - nonholiday_mean) / max(abs(nonholiday_mean), 1e-6)) if not math.isnan(holiday_mean) else math.nan,
            "hourly_profile_range_pct": float(100.0 * (by_hour.max() - by_hour.min()) / max(abs(by_hour.mean()), 1e-6)),
            "monthly_profile_range_pct": float(100.0 * (by_month.max() - by_month.min()) / max(abs(by_month.mean()), 1e-6)),
        })
    return pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)


def compute_autocorrelation_table(
    df: pd.DataFrame,
    zone_cols: Sequence[str],
    max_lag: int,
    selected_lags: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    full_rows = []
    selected_rows = []
    for zone in zone_cols:
        s = pd.to_numeric(df[zone], errors="coerce")
        for lag in range(1, max_lag + 1):
            ac = float(s.autocorr(lag=lag))
            full_rows.append({"zone": zone, "lag_hours": lag, "autocorr": ac})
            if lag in selected_lags:
                selected_rows.append({"zone": zone, "lag_hours": lag, "autocorr": ac})
    return pd.DataFrame(full_rows), pd.DataFrame(selected_rows)


def compute_ramp_stats(df: pd.DataFrame, zone_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for zone in zone_cols:
        s = pd.to_numeric(df[zone], errors="coerce")
        d1 = s.diff()
        ad1 = d1.abs()
        rows.append({
            "zone": zone,
            "mean_abs_1h_ramp": float(ad1.mean()),
            "p90_abs_1h_ramp": float(ad1.quantile(0.90)),
            "p95_abs_1h_ramp": float(ad1.quantile(0.95)),
            "p99_abs_1h_ramp": float(ad1.quantile(0.99)),
            "max_abs_1h_ramp": float(ad1.max()),
            "std_1h_ramp": float(d1.std()),
        })
    return pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)


def compute_horizon_baselines(
    energy_df: pd.DataFrame,
    zone_cols: Sequence[str],
    eval_years: Sequence[int],
    history_len: int,
    future_len: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate simple baselines on 24-hour windows.
    """
    values = energy_df[list(zone_cols)].to_numpy(dtype=np.float64)
    idx = energy_df.index

    midnight_mask = (idx.year.isin(eval_years)) & (idx.hour == 0)
    starts = np.where(np.asarray(midnight_mask))[0]
    starts = starts[(starts >= history_len) & (starts + future_len <= len(idx))]

    model_names = [
        "prev_day_same_hour",
        "prev_week_same_hour",
        "blend_day_week",
        "last_value_repeat",
    ]

    overall_rows = []
    zone_rows = []
    horizon_rows = []

    for model_name in model_names:
        abs_pct_errors_overall = []
        zone_error_store = {z: [] for z in zone_cols}
        horizon_error_store = {h: [] for h in range(1, future_len + 1)}

        for t in starts:
            target = values[t : t + future_len]

            if model_name == "prev_day_same_hour":
                pred = values[t - 24 : t]
            elif model_name == "prev_week_same_hour":
                pred = values[t - 168 : t - 144]
            elif model_name == "blend_day_week":
                pred = 0.5 * values[t - 24 : t] + 0.5 * values[t - 168 : t - 144]
            elif model_name == "last_value_repeat":
                pred = np.repeat(values[t - 1 : t], repeats=future_len, axis=0)
            else:
                raise ValueError(model_name)

            denom = np.maximum(np.abs(target), 1e-6)
            ape = np.abs(pred - target) / denom
            abs_pct_errors_overall.append(ape.reshape(-1))

            for zi, zone in enumerate(zone_cols):
                zone_error_store[zone].append(ape[:, zi])
            for h in range(future_len):
                horizon_error_store[h + 1].append(ape[h, :])

        if len(abs_pct_errors_overall) == 0:
            continue

        overall_rows.append({
            "eval_years": ",".join(map(str, eval_years)),
            "baseline": model_name,
            "num_windows": int(len(starts)),
            "overall_mape": float(np.mean(np.concatenate(abs_pct_errors_overall)) * 100.0),
        })

        for zone in zone_cols:
            zone_rows.append({
                "eval_years": ",".join(map(str, eval_years)),
                "baseline": model_name,
                "zone": zone,
                "mape": float(np.mean(np.concatenate(zone_error_store[zone])) * 100.0),
            })

        for h in range(1, future_len + 1):
            horizon_rows.append({
                "eval_years": ",".join(map(str, eval_years)),
                "baseline": model_name,
                "horizon_hour": h,
                "mape": float(np.mean(np.concatenate(horizon_error_store[h])) * 100.0),
            })

    return pd.DataFrame(overall_rows), pd.DataFrame(zone_rows), pd.DataFrame(horizon_rows)


def sample_weather_timeseries(
    energy_df: pd.DataFrame,
    zone_cols: Sequence[str],
    weather_lookup: Dict[str, Path],
    weather_var_names: Sequence[str],
    weather_stride: int,
    max_weather_files: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    sampled_count = 0

    for i, ts in enumerate(tqdm(energy_df.index, desc="Sampling weather tensors")):
        if weather_stride > 1 and (i % weather_stride) != 0:
            continue
        key = ts.strftime("%Y%m%d%H")
        if key not in weather_lookup:
            continue
        x = safe_torch_load(weather_lookup[key])
        if x.ndim != 3 or x.shape[-1] < 1:
            continue

        x_mean = x.mean(dim=(0, 1)).cpu().numpy()
        x_std = x.std(dim=(0, 1)).cpu().numpy()
        x_min = x.amin(dim=(0, 1)).cpu().numpy()
        x_max = x.amax(dim=(0, 1)).cpu().numpy()

        rec: Dict[str, float] = {"timestamp": ts}
        for zi, zone in enumerate(zone_cols):
            rec[f"target__{zone}"] = float(energy_df.iloc[i][zone])
        rec["target__total"] = float(energy_df.iloc[i][list(zone_cols)].sum())

        n_channels = x.shape[-1]
        for c in range(n_channels):
            cname = weather_var_names[c] if c < len(weather_var_names) else f"ch_{c}"
            rec[f"mean__{cname}"] = float(x_mean[c])
            rec[f"std__{cname}"] = float(x_std[c])
            rec[f"min__{cname}"] = float(x_min[c])
            rec[f"max__{cname}"] = float(x_max[c])
        records.append(rec)
        sampled_count += 1
        if max_weather_files is not None and sampled_count >= max_weather_files:
            break

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    sampled_df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

    stats_rows = []
    corr_rows = []

    mean_cols = sorted([c for c in sampled_df.columns if c.startswith("mean__")])
    for mean_col in mean_cols:
        cname = mean_col.split("mean__", 1)[1]
        std_col = f"std__{cname}"
        min_col = f"min__{cname}"
        max_col = f"max__{cname}"

        stats_rows.append({
            "channel": cname,
            "sample_count": int(len(sampled_df)),
            "temporal_mean_of_spatial_mean": float(sampled_df[mean_col].mean()),
            "temporal_std_of_spatial_mean": float(sampled_df[mean_col].std()),
            "mean_spatial_std": float(sampled_df[std_col].mean()) if std_col in sampled_df else math.nan,
            "global_min_over_samples": float(sampled_df[min_col].min()) if min_col in sampled_df else math.nan,
            "global_max_over_samples": float(sampled_df[max_col].max()) if max_col in sampled_df else math.nan,
        })

        for zone in list(zone_cols) + ["total"]:
            target_col = f"target__{zone}"
            if target_col not in sampled_df:
                continue
            corr_mean = sampled_df[[mean_col, target_col]].corr().iloc[0, 1]
            corr_rows.append({
                "channel": cname,
                "feature_type": "spatial_mean",
                "target": zone,
                "corr": float(corr_mean),
            })
            if std_col in sampled_df:
                corr_std = sampled_df[[std_col, target_col]].corr().iloc[0, 1]
                corr_rows.append({
                    "channel": cname,
                    "feature_type": "spatial_std",
                    "target": zone,
                    "corr": float(corr_std),
                })

    return pd.DataFrame(stats_rows), pd.DataFrame(corr_rows)


def compute_split_summary(df: pd.DataFrame, train_years: Sequence[int], val_years: Sequence[int], test_years: Sequence[int]) -> pd.DataFrame:
    rows = []
    for split_name, years in [
        ("train", train_years),
        ("val", val_years),
        ("test", test_years),
        ("all", sorted(set(train_years) | set(val_years) | set(test_years))),
    ]:
        if len(years) == 0:
            continue
        mask = df.index.year.isin(years)
        sdf = df.loc[mask]
        rows.append({
            "split": split_name,
            "years": ",".join(map(str, years)),
            "num_rows": int(len(sdf)),
            "start": str(sdf.index.min()) if len(sdf) else None,
            "end": str(sdf.index.max()) if len(sdf) else None,
        })
    return pd.DataFrame(rows)


# ============================================================
# Plotting helpers for dataset profiling
# ============================================================
def plot_zone_correlation(corr_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_df.values, aspect="auto")
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_title("Zone-to-zone demand correlation")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_hourly_profiles(hourly_df: pd.DataFrame, zone_cols: Sequence[str], out_path: Path) -> None:
    pivot = hourly_df.pivot(index="hour_local", columns="zone", values="mean_load")
    fig, ax = plt.subplots(figsize=(10, 5))
    for zone in zone_cols:
        if zone in pivot.columns:
            ax.plot(pivot.index.values, pivot[zone].values, label=zone)
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Mean load")
    ax.set_title("Mean demand by local hour")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_acf_selected(acf_selected_df: pd.DataFrame, out_path: Path) -> None:
    if acf_selected_df.empty:
        return
    pivot = acf_selected_df.pivot(index="lag_hours", columns="zone", values="autocorr")
    fig, ax = plt.subplots(figsize=(9, 5))
    for zone in pivot.columns:
        ax.plot(pivot.index.values, pivot[zone].values, marker="o", label=zone)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Selected lag autocorrelation by zone")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_baseline_horizon(horizon_df: pd.DataFrame, out_path: Path) -> None:
    if horizon_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for baseline in sorted(horizon_df["baseline"].unique()):
        sdf = horizon_df[horizon_df["baseline"] == baseline].sort_values("horizon_hour")
        ax.plot(sdf["horizon_hour"].values, sdf["mape"].values, marker="o", label=baseline)
    ax.set_xlabel("Forecast horizon hour")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Naive baseline MAPE by forecast horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ============================================================
# Recommendation text from dataset profiling
# ============================================================
def build_recommendations(
    zone_stats_df: pd.DataFrame,
    regime_effect_df: pd.DataFrame,
    acf_selected_df: pd.DataFrame,
    zone_corr_df: pd.DataFrame,
    baseline_overall_df: pd.DataFrame,
    weather_corr_df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("Dataset-analysis-driven architecture guidance")
    lines.append("=" * 54)
    lines.append("")

    mean_cv = float(zone_stats_df["cv"].mean()) if not zone_stats_df.empty else math.nan
    if not math.isnan(mean_cv):
        lines.append(f"Average per-zone coefficient of variation: {mean_cv:.4f}")
        if mean_cv > 0.18:
            lines.append("- Zones have materially different scales/volatility. Add zone embeddings or zone-specific heads/queries.")
        else:
            lines.append("- Zone scales are relatively aligned. A shared backbone is appropriate, but zone-conditioned outputs can still help.")
        lines.append("")

    mean_abs_zone_corr = math.nan
    if not zone_corr_df.empty:
        zc = zone_corr_df.to_numpy(dtype=float)
        mask = ~np.eye(zc.shape[0], dtype=bool)
        mean_abs_zone_corr = float(np.nanmean(np.abs(zc[mask])))
        lines.append(f"Mean absolute cross-zone correlation: {mean_abs_zone_corr:.4f}")
        if mean_abs_zone_corr >= 0.70:
            lines.append("- Strong shared system dynamics. Keep a shared encoder, but decode with zone queries or zone-specific heads instead of one shared summary vector.")
        elif mean_abs_zone_corr >= 0.40:
            lines.append("- Moderate cross-zone coupling. Shared trunk + light zone personalization is a good tradeoff.")
        else:
            lines.append("- Weak cross-zone coupling. Consider more zone-specific capacity.")
        lines.append("")

    if not regime_effect_df.empty:
        weekend_shift = float(np.nanmean(np.abs(regime_effect_df["weekend_vs_weekday_pct"])))
        monthly_shift = float(np.nanmean(np.abs(regime_effect_df["monthly_profile_range_pct"])))
        hourly_shift = float(np.nanmean(np.abs(regime_effect_df["hourly_profile_range_pct"])))
        lines.append(f"Average |weekend-weekday| shift (%): {weekend_shift:.2f}")
        lines.append(f"Average hourly profile range (% of mean): {hourly_shift:.2f}")
        lines.append(f"Average monthly profile range (% of mean): {monthly_shift:.2f}")
        if weekend_shift > 5.0 or monthly_shift > 10.0 or hourly_shift > 10.0:
            lines.append("- Calendar structure is strong. Expand beyond sin/cos hour + day. Add holiday, weekend, month/season, and preferably local-time features/embeddings.")
        else:
            lines.append("- Calendar effects look modest. Basic cyclical features may already cover much of the signal.")
        lines.append("")

    if not acf_selected_df.empty:
        def lag_mean(lag: int) -> float:
            s = acf_selected_df.loc[acf_selected_df["lag_hours"] == lag, "autocorr"]
            return float(s.mean()) if len(s) else math.nan

        lag24 = lag_mean(24)
        lag168 = lag_mean(168)
        lag1 = lag_mean(1)
        lines.append(f"Mean autocorrelation at 1h / 24h / 168h: {lag1:.4f}, {lag24:.4f}, {lag168:.4f}")
        if not math.isnan(lag168) and lag168 > 0.60:
            lines.append("- Strong weekly periodicity. Use longer history, a previous-week residual baseline, or multi-resolution history encoding.")
        if not math.isnan(lag24) and lag24 > 0.70:
            lines.append("- Strong daily periodicity. Previous-day residual baselines and explicit daily features should help.")
        if not math.isnan(lag1) and lag1 > 0.90 and (math.isnan(lag24) or lag24 < lag1):
            lines.append("- Very strong local persistence. Include a demand-only or residual shortcut branch.")
        lines.append("")

    if not baseline_overall_df.empty:
        best_row = baseline_overall_df.sort_values("overall_mape", ascending=True).iloc[0]
        lines.append(f"Best naive baseline: {best_row['baseline']} with MAPE {best_row['overall_mape']:.3f}%")
        if str(best_row["baseline"]) in {"prev_day_same_hour", "prev_week_same_hour", "blend_day_week"}:
            lines.append("- A simple seasonal baseline is already strong. The next architecture should predict residual corrections over that baseline, not raw load from scratch.")
        else:
            lines.append("- Pure persistence is strongest. A direct autoregressive shortcut is worth including.")
        lines.append("")

    if not weather_corr_df.empty:
        tmp = weather_corr_df.copy()
        tmp["abs_corr"] = tmp["corr"].abs()
        best_weather = tmp.sort_values("abs_corr", ascending=False).iloc[0]
        lines.append(
            f"Strongest sampled weather-demand correlation: channel={best_weather['channel']}, "
            f"feature={best_weather['feature_type']}, target={best_weather['target']}, |corr|={best_weather['abs_corr']:.4f}"
        )
        if float(best_weather["abs_corr"]) >= 0.30:
            lines.append("- Weather has measurable signal even after crude spatial aggregation. Keep the weather branch and consider better spatial conditioning or a stronger fusion module.")
        else:
            lines.append("- Weather signal may be subtle or highly non-linear. Before enlarging the weather encoder, test whether demand/history features are currently dominating performance.")
        lines.append("")

    lines.append("Suggested experiment order")
    lines.append("- 1) Fix evaluation breadth and report horizon/zone-level metrics.")
    lines.append("- 2) Compare 24h vs 48h vs 168h history, preferably with a residual baseline branch.")
    lines.append("- 3) Add richer calendar/local-time embeddings.")
    lines.append("- 4) Normalize weather channels and test stronger weather fusion only if sampled weather correlations justify it.")
    lines.append("- 5) Replace the single shared future summary with zone-aware decoding if cross-zone correlation is high but zone scales differ.")
    return "\n".join(lines) + "\n"


# ============================================================
# Inference-only checkpoint analysis
# ============================================================
@dataclass
class EvalPayload:
    preds: np.ndarray                # (N, F, Z)
    targets: np.ndarray              # (N, F, Z)
    baseline_prev_day: np.ndarray    # (N, F, Z)
    forecast_hours: np.ndarray       # (N, F) int64 unix hours
    start_indices: np.ndarray        # (N,)
    zone_names: List[str]
    eval_years: List[int]
    ablation_name: str


def dynamic_import(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_from_path(
    model_path: Path,
    metadata: Dict[str, object],
    device: torch.device,
    checkpoint_override: Optional[Path] = None,
):
    module = dynamic_import("analysis_model_module", model_path)

    model_dir = str(model_path.parent.resolve())
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    if not hasattr(module, "get_model"):
        raise AttributeError(f"{model_path} does not expose get_model(metadata)")
    model = module.get_model(metadata)

    if checkpoint_override is not None and checkpoint_override.exists():
        ckpt = torch.load(checkpoint_override, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt, strict=False)
        else:
            raise TypeError("Unsupported checkpoint format for checkpoint override.")

    model = model.to(device)
    model.eval()
    return model


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_eval_start_indices(
    energy_df: pd.DataFrame,
    eval_years: Sequence[int],
    history_len: int,
    future_len: int,
    midnight_only: bool = True,
    max_windows: Optional[int] = None,
) -> np.ndarray:
    idx = energy_df.index
    if midnight_only:
        mask = (idx.year.isin(eval_years)) & (idx.hour == 0)
    else:
        mask = idx.year.isin(eval_years)
    starts = np.where(np.asarray(mask))[0]
    starts = starts[(starts >= history_len) & (starts + future_len <= len(idx))]
    if max_windows is not None and len(starts) > max_windows:
        starts = starts[-max_windows:]
    return starts.astype(np.int64)


def load_weather_cached(hour_int64: int, weather_lookup: Dict[str, Path], cache: Dict[int, torch.Tensor]) -> torch.Tensor:
    if hour_int64 in cache:
        return cache[hour_int64]
    dt = pd.Timestamp(int(hour_int64), unit="h", tz="UTC")
    key = dt.strftime("%Y%m%d%H")
    if key not in weather_lookup:
        raise FileNotFoundError(f"Missing weather file for {dt}")
    t = safe_torch_load(weather_lookup[key])
    cache[hour_int64] = t
    if len(cache) > 512:
        oldest = next(iter(cache))
        del cache[oldest]
    return t


def get_model_used_history(model, raw_history_len: int) -> int:
    """
    Determine the number of raw history hours actually used by the model.
    For the current project model this is model.S.
    """
    used = getattr(model, "S", None)
    if used is None:
        return raw_history_len
    try:
        used = int(used)
    except Exception:
        return raw_history_len
    return min(max(1, used), raw_history_len)


def apply_raw_input_ablation(
    ablation_name: str,
    hist_weather: torch.Tensor,
    hist_energy: torch.Tensor,
    fut_weather: torch.Tensor,
    fut_time: torch.Tensor,
    model,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Operates on the raw inputs passed to model.adapt_inputs(...).
    Shapes:
      hist_weather: (1, H, 450, 449, C)
      hist_energy : (1, H, Z)
      fut_weather : (1, F, 450, 449, C)
      fut_time    : (1, F)
    """
    hw = hist_weather.clone()
    he = hist_energy.clone()
    fw = fut_weather.clone()
    ft = fut_time.clone()

    if ablation_name == "none":
        return hw, he, fw, ft

    if ablation_name == "zero_future_weather":
        fw.zero_()
        return hw, he, fw, ft

    if ablation_name == "zero_history_weather":
        used = get_model_used_history(model, hw.shape[1])
        hw[:, -used:] = 0.0
        return hw, he, fw, ft

    if ablation_name == "zero_all_weather":
        hw.zero_()
        fw.zero_()
        return hw, he, fw, ft

    if ablation_name == "zero_history_energy":
        used = get_model_used_history(model, he.shape[1])
        he[:, -used:] = 0.0
        return hw, he, fw, ft

    if ablation_name.startswith("history_keep_last_") and ablation_name.endswith("h"):
        used = get_model_used_history(model, he.shape[1])
        keep = int(ablation_name.split("history_keep_last_", 1)[1][:-1])
        keep = max(1, min(keep, used))
        drop = used - keep
        if drop > 0:
            hw[:, -(used):-(keep)] = 0.0
            he[:, -(used):-(keep)] = 0.0
        return hw, he, fw, ft

    if ablation_name.startswith("zero_weather_channel_"):
        ch = int(ablation_name.split("zero_weather_channel_", 1)[1])
        if ch < hw.shape[-1]:
            hw[..., ch] = 0.0
            fw[..., ch] = 0.0
        return hw, he, fw, ft

    # These ablations act only after adapt_inputs(), because this model constructs
    # calendar features inside adapt_inputs rather than receiving them as raw inputs.
    if ablation_name in {"zero_calendar", "zero_history_calendar", "zero_future_calendar"}:
        return hw, he, fw, ft

    raise ValueError(f"Unknown raw-input ablation: {ablation_name}")


def apply_adapted_ablation(ablation_name: str, adapted_inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    """
    Operates after model.adapt_inputs(...). This is useful for calendar ablations because
    the current model creates calendar features inside adapt_inputs.
    Expected adapted output for the current model:
      weather_seq, hist_demand, hist_cal, fut_cal
    """
    if ablation_name == "none":
        return adapted_inputs

    items = list(adapted_inputs)

    if ablation_name == "zero_calendar":
        if len(items) >= 4:
            items[2] = torch.zeros_like(items[2])
            items[3] = torch.zeros_like(items[3])
        return tuple(items)

    if ablation_name == "zero_history_calendar":
        if len(items) >= 4:
            items[2] = torch.zeros_like(items[2])
        return tuple(items)

    if ablation_name == "zero_future_calendar":
        if len(items) >= 4:
            items[3] = torch.zeros_like(items[3])
        return tuple(items)

    return adapted_inputs


def run_model_eval(
    model,
    energy_df: pd.DataFrame,
    weather_lookup: Dict[str, Path],
    zone_cols: Sequence[str],
    eval_years: Sequence[int],
    raw_history_len: int,
    future_len: int,
    ablation_name: str,
    device: torch.device,
    max_windows: Optional[int] = None,
    midnight_only: bool = True,
    log_every: int = 25,
) -> EvalPayload:
    starts = build_eval_start_indices(
        energy_df=energy_df,
        eval_years=eval_years,
        history_len=raw_history_len,
        future_len=future_len,
        midnight_only=midnight_only,
        max_windows=max_windows,
    )
    if len(starts) == 0:
        raise RuntimeError(f"No eval windows found for years {eval_years}")

    energy_values = energy_df[list(zone_cols)].to_numpy(dtype=np.float32)
    all_hours = energy_df.index.values.astype("datetime64[h]").astype(np.int64)
    weather_cache: Dict[int, torch.Tensor] = {}

    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    prev_day_list: List[np.ndarray] = []
    fut_hours_list: List[np.ndarray] = []

    if hasattr(torch, "set_grad_enabled"):
        torch.set_grad_enabled(False)

    with torch.no_grad():
        for step, t_idx in enumerate(tqdm(starts, desc=f"Running {ablation_name}")):
            hist_slice = slice(t_idx - raw_history_len, t_idx)
            future_slice = slice(t_idx, t_idx + future_len)

            hist_hours_arr = all_hours[hist_slice]
            future_hours_arr = all_hours[future_slice]

            hist_energy = energy_values[hist_slice]
            target = energy_values[future_slice]
            prev_day = energy_values[t_idx - 24 : t_idx]

            hist_weather = torch.stack([
                load_weather_cached(int(h), weather_lookup, weather_cache) for h in hist_hours_arr
            ], dim=0)
            fut_weather = torch.stack([
                load_weather_cached(int(h), weather_lookup, weather_cache) for h in future_hours_arr
            ], dim=0)

            raw_inputs = (
                hist_weather.unsqueeze(0).to(device),                    # (1, H, Hx, Wx, C)
                torch.from_numpy(hist_energy).unsqueeze(0).to(device),  # (1, H, Z)
                fut_weather.unsqueeze(0).to(device),                    # (1, F, Hx, Wx, C)
                torch.tensor(future_hours_arr, dtype=torch.int64, device=device).unsqueeze(0),  # (1, F)
            )

            raw_inputs = apply_raw_input_ablation(ablation_name, *raw_inputs, model=model)
            adapted = model.adapt_inputs(*raw_inputs)
            adapted = apply_adapted_ablation(ablation_name, adapted)
            pred = model(*adapted).squeeze(0).detach().cpu().numpy()

            preds_list.append(pred.astype(np.float32))
            targets_list.append(target.astype(np.float32))
            prev_day_list.append(prev_day.astype(np.float32))
            fut_hours_list.append(future_hours_arr.astype(np.int64))

            if log_every > 0 and ((step + 1) % log_every == 0 or (step + 1) == len(starts)):
                cur_pred = np.stack(preds_list, axis=0)
                cur_tgt = np.stack(targets_list, axis=0)
                running_mape = mape_np(cur_pred, cur_tgt)
                print(f"[{ablation_name}] processed {step+1}/{len(starts)} windows | running overall MAPE={running_mape:.4f}%")

    return EvalPayload(
        preds=np.stack(preds_list, axis=0),
        targets=np.stack(targets_list, axis=0),
        baseline_prev_day=np.stack(prev_day_list, axis=0),
        forecast_hours=np.stack(fut_hours_list, axis=0),
        start_indices=starts,
        zone_names=list(zone_cols),
        eval_years=list(eval_years),
        ablation_name=ablation_name,
    )


def payload_to_long_df(payload: EvalPayload, energy_df_full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    zone_names = payload.zone_names
    n_windows, future_len, n_zones = payload.targets.shape
    for wi in range(n_windows):
        for h in range(future_len):
            ts_utc = pd.Timestamp(int(payload.forecast_hours[wi, h]), unit="h", tz="UTC")
            ts_local = ts_utc.tz_convert("US/Eastern")
            for zi, zone in enumerate(zone_names):
                y = float(payload.targets[wi, h, zi])
                p = float(payload.preds[wi, h, zi])
                b = float(payload.baseline_prev_day[wi, h, zi])

                t_idx = int(payload.start_indices[wi] + h)
                prev_val = float(energy_df_full.iloc[t_idx - 1][zone]) if t_idx - 1 >= 0 else math.nan
                ramp = y - prev_val if not math.isnan(prev_val) else math.nan
                abs_ramp = abs(ramp) if not math.isnan(ramp) else math.nan

                rows.append({
                    "window_idx": wi,
                    "forecast_horizon": h + 1,
                    "timestamp_utc": ts_utc,
                    "timestamp_local": ts_local,
                    "zone": zone,
                    "target": y,
                    "pred": p,
                    "prev_day_baseline": b,
                    "abs_pct_error": abs(p - y) / max(abs(y), 1e-6) * 100.0,
                    "baseline_abs_pct_error": abs(b - y) / max(abs(y), 1e-6) * 100.0,
                    "error": p - y,
                    "abs_error": abs(p - y),
                    "sq_error": (p - y) ** 2,
                    "hour_local": ts_local.hour,
                    "dow_local": ts_local.dayofweek,
                    "is_weekend": int(ts_local.dayofweek >= 5),
                    "month_local": ts_local.month,
                    "target_prev_hour": prev_val,
                    "ramp_1h": ramp,
                    "abs_ramp_1h": abs_ramp,
                })
    long_df = pd.DataFrame(rows)

    if not long_df.empty:
        # global quantiles across all zone-horizon rows
        long_df["target_quantile_bin"] = pd.qcut(
            long_df["target"].rank(method="first"),
            q=5,
            labels=[f"Q{i}" for i in range(1, 6)],
        )
        valid_ramps = long_df["abs_ramp_1h"].fillna(0.0)
        long_df["abs_ramp_quantile_bin"] = pd.qcut(
            valid_ramps.rank(method="first"),
            q=5,
            labels=[f"Q{i}" for i in range(1, 6)],
        )
    return long_df


def aggregate_metric_table(long_df: pd.DataFrame, group_cols: Sequence[str], prefix: str) -> pd.DataFrame:
    rows = []
    if long_df.empty:
        return pd.DataFrame()

    g = long_df.groupby(list(group_cols), dropna=False)
    for key, sdf in g:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        pred = sdf["pred"].to_numpy(dtype=np.float64)
        tgt = sdf["target"].to_numpy(dtype=np.float64)
        base = sdf["prev_day_baseline"].to_numpy(dtype=np.float64)
        row.update({
            "count": int(len(sdf)),
            "mape": mape_np(pred, tgt),
            "smape": smape_np(pred, tgt),
            "mae": mae_np(pred, tgt),
            "rmse": rmse_np(pred, tgt),
            "bias": bias_np(pred, tgt),
            "baseline_prev_day_mape": mape_np(base, tgt),
            "mape_delta_vs_prev_day": mape_np(pred, tgt) - mape_np(base, tgt),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and len(group_cols) > 0:
        out = out.sort_values(list(group_cols)).reset_index(drop=True)
    return out


def save_payload_metrics(payload: EvalPayload, energy_df_full: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    ensure_dir(out_dir)
    long_df = payload_to_long_df(payload, energy_df_full)
    long_df.to_csv(out_dir / "forecast_row_level_metrics.csv", index=False)

    overall_df = aggregate_metric_table(long_df.assign(dummy="all"), ["dummy"], prefix="overall")
    if not overall_df.empty:
        overall_df = overall_df.drop(columns=["dummy"])
    overall_df.insert(0, "ablation", payload.ablation_name)
    overall_df.insert(1, "eval_years", ",".join(map(str, payload.eval_years)))
    overall_df.to_csv(out_dir / "overall_metrics.csv", index=False)

    by_zone_df = aggregate_metric_table(long_df, ["zone"], prefix="zone")
    by_zone_df.insert(0, "ablation", payload.ablation_name)
    by_zone_df.to_csv(out_dir / "metrics_by_zone.csv", index=False)

    by_horizon_df = aggregate_metric_table(long_df, ["forecast_horizon"], prefix="horizon")
    by_horizon_df.insert(0, "ablation", payload.ablation_name)
    by_horizon_df.to_csv(out_dir / "metrics_by_horizon.csv", index=False)

    by_hour_df = aggregate_metric_table(long_df, ["hour_local"], prefix="hour_local")
    by_hour_df.insert(0, "ablation", payload.ablation_name)
    by_hour_df.to_csv(out_dir / "metrics_by_local_hour.csv", index=False)

    by_weekend_df = aggregate_metric_table(long_df, ["is_weekend"], prefix="weekend")
    by_weekend_df.insert(0, "ablation", payload.ablation_name)
    by_weekend_df.to_csv(out_dir / "metrics_by_weekend.csv", index=False)

    by_target_quant_df = aggregate_metric_table(long_df, ["target_quantile_bin"], prefix="target_quantile")
    by_target_quant_df.insert(0, "ablation", payload.ablation_name)
    by_target_quant_df.to_csv(out_dir / "metrics_by_target_quantile.csv", index=False)

    by_ramp_quant_df = aggregate_metric_table(long_df, ["abs_ramp_quantile_bin"], prefix="ramp_quantile")
    by_ramp_quant_df.insert(0, "ablation", payload.ablation_name)
    by_ramp_quant_df.to_csv(out_dir / "metrics_by_abs_ramp_quantile.csv", index=False)

    return long_df


def plot_model_vs_baseline_horizon(long_df: pd.DataFrame, title: str, out_path: Path) -> None:
    if long_df.empty:
        return
    by_h = long_df.groupby("forecast_horizon", as_index=False).agg({
        "abs_pct_error": "mean",
        "baseline_abs_pct_error": "mean",
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(by_h["forecast_horizon"], by_h["abs_pct_error"], marker="o", label="model")
    ax.plot(by_h["forecast_horizon"], by_h["baseline_abs_pct_error"], marker="o", label="prev_day_baseline")
    ax.set_xlabel("Forecast horizon hour")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_model_vs_baseline_zone(long_df: pd.DataFrame, title: str, out_path: Path) -> None:
    if long_df.empty:
        return
    by_z = long_df.groupby("zone", as_index=False).agg({
        "abs_pct_error": "mean",
        "baseline_abs_pct_error": "mean",
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(by_z))
    width = 0.38
    ax.bar(x - width / 2, by_z["abs_pct_error"], width=width, label="model")
    ax.bar(x + width / 2, by_z["baseline_abs_pct_error"], width=width, label="prev_day_baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(by_z["zone"])
    ax.set_ylabel("MAPE (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_ablation_overall(ablation_summary_df: pd.DataFrame, out_path: Path) -> None:
    if ablation_summary_df.empty:
        return
    df = ablation_summary_df.sort_values("overall_mape")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(df["ablation"], df["overall_mape"])
    ax.set_ylabel("Overall MAPE (%)")
    ax.set_title("Inference-only ablation comparison")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_ablation_horizon(ablation_horizon_df: pd.DataFrame, out_path: Path) -> None:
    if ablation_horizon_df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    for ablation in ablation_horizon_df["ablation"].unique():
        sdf = ablation_horizon_df[ablation_horizon_df["ablation"] == ablation].sort_values("forecast_horizon")
        ax.plot(sdf["forecast_horizon"], sdf["mape"], marker="o", label=ablation)
    ax.set_xlabel("Forecast horizon hour")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Inference-only ablations by horizon")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_ablation_list(model, run_channel_ablation: bool, history_keep_hours: Sequence[int], n_weather_vars: int) -> List[str]:
    ablations = [
        "none",
        "zero_future_weather",
        "zero_history_weather",
        "zero_all_weather",
        "zero_history_energy",
        "zero_calendar",
    ]
    used = get_model_used_history(model, 168)
    hist_hours = [h for h in history_keep_hours if 1 <= int(h) < used]
    for h in hist_hours:
        ablations.append(f"history_keep_last_{int(h)}h")
    if run_channel_ablation:
        for ch in range(int(n_weather_vars)):
            ablations.append(f"zero_weather_channel_{ch}")
    return ablations


def fit_blend_weights(
    calib_df: pd.DataFrame,
    grid: np.ndarray,
    mode: str,
) -> pd.DataFrame:
    """
    mode:
      - global
      - by_zone
      - by_zone_horizon
    Fits alpha in:
      final = alpha * model + (1-alpha) * prev_day_baseline
    using calibration data and MAPE objective over the provided groups.
    """
    if calib_df.empty:
        return pd.DataFrame()

    if mode == "global":
        group_cols = ["dummy"]
        work = calib_df.assign(dummy="all")
    elif mode == "by_zone":
        group_cols = ["zone"]
        work = calib_df.copy()
    elif mode == "by_zone_horizon":
        group_cols = ["zone", "forecast_horizon"]
        work = calib_df.copy()
    else:
        raise ValueError(mode)

    rows = []
    for key, sdf in work.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        pred = sdf["pred"].to_numpy(dtype=np.float64)
        base = sdf["prev_day_baseline"].to_numpy(dtype=np.float64)
        tgt = sdf["target"].to_numpy(dtype=np.float64)

        best_alpha = None
        best_mape = None
        for alpha in grid:
            final = alpha * pred + (1.0 - alpha) * base
            m = mape_np(final, tgt)
            if best_mape is None or m < best_mape:
                best_mape = m
                best_alpha = float(alpha)

        row = {col: val for col, val in zip(group_cols, key)}
        row["alpha"] = float(best_alpha)
        row["calib_mape"] = float(best_mape)
        row["count"] = int(len(sdf))
        rows.append(row)

    out = pd.DataFrame(rows)
    if "dummy" in out.columns:
        out = out.drop(columns=["dummy"])
    return out


def apply_blend_weights(report_df: pd.DataFrame, weights_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if report_df.empty or weights_df.empty:
        return pd.DataFrame()
    if mode == "global":
        alpha = float(weights_df["alpha"].iloc[0])
        out = report_df.copy()
        out["blend_pred"] = alpha * out["pred"] + (1.0 - alpha) * out["prev_day_baseline"]
        return out
    if mode == "by_zone":
        out = report_df.merge(weights_df[["zone", "alpha"]], on="zone", how="left")
        out["alpha"] = out["alpha"].fillna(1.0)
        out["blend_pred"] = out["alpha"] * out["pred"] + (1.0 - out["alpha"]) * out["prev_day_baseline"]
        return out
    if mode == "by_zone_horizon":
        out = report_df.merge(weights_df[["zone", "forecast_horizon", "alpha"]], on=["zone", "forecast_horizon"], how="left")
        out["alpha"] = out["alpha"].fillna(1.0)
        out["blend_pred"] = out["alpha"] * out["pred"] + (1.0 - out["alpha"]) * out["prev_day_baseline"]
        return out
    raise ValueError(mode)


def summarize_blend_df(blend_df: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if blend_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    pred = blend_df["blend_pred"].to_numpy(dtype=np.float64)
    tgt = blend_df["target"].to_numpy(dtype=np.float64)
    overall = pd.DataFrame([{
        "blend_label": label,
        "count": int(len(blend_df)),
        "mape": mape_np(pred, tgt),
        "smape": smape_np(pred, tgt),
        "mae": mae_np(pred, tgt),
        "rmse": rmse_np(pred, tgt),
        "bias": bias_np(pred, tgt),
    }])
    by_h = []
    for h, sdf in blend_df.groupby("forecast_horizon"):
        by_h.append({
            "blend_label": label,
            "forecast_horizon": int(h),
            "mape": mape_np(sdf["blend_pred"].to_numpy(dtype=np.float64), sdf["target"].to_numpy(dtype=np.float64)),
        })
    return overall, pd.DataFrame(by_h).sort_values("forecast_horizon").reset_index(drop=True)


def write_inference_summary_text(
    ablation_summary_df: pd.DataFrame,
    blend_overall_df: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("Inference-only analysis summary")
    lines.append("=" * 40)
    lines.append("")
    if not ablation_summary_df.empty:
        best = ablation_summary_df.sort_values("overall_mape").iloc[0]
        base = ablation_summary_df.loc[ablation_summary_df["ablation"] == "none"]
        base_mape = float(base["overall_mape"].iloc[0]) if len(base) else math.nan
        lines.append(f"Best ablation setting by raw evaluation MAPE: {best['ablation']} ({best['overall_mape']:.4f}%)")
        if not math.isnan(base_mape):
            lines.append(f"Reference trained model MAPE under standard inference: {base_mape:.4f}%")
        lines.append("")
        lines.append("Ablation interpretation rules of thumb:")
        lines.append("- Large degradation from zero_future_weather => future weather branch is carrying real predictive signal.")
        lines.append("- Large degradation from zero_history_energy => demand-history branch is dominant / essential.")
        lines.append("- Small degradation from zero_calendar => calendar branch may be underused or redundant with autoregressive structure.")
        lines.append("- Small degradation from zero_history_weather => historical weather branch may be weak relative to future weather.")
        lines.append("- Strong changes under history_keep_last_kh => the current model is sensitive to recent memory length within its trained window.")
        lines.append("")

    if not blend_overall_df.empty:
        best_blend = blend_overall_df.sort_values("mape").iloc[0]
        lines.append(f"Best no-retraining blend on held-out report windows: {best_blend['blend_label']} ({best_blend['mape']:.4f}%)")
        lines.append("- If a blend beats the raw model clearly, the next architecture should likely be residual-over-baseline rather than direct raw-load prediction.")
        lines.append("- If per-zone-horizon blending helps most, the model likely needs a more zone-aware and horizon-aware output head.")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def read_existing_ablation_outputs(ablation_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_path = ablation_dir / "overall_metrics.csv"
    by_horizon_path = ablation_dir / "metrics_by_horizon.csv"
    row_level_path = ablation_dir / "forecast_row_level_metrics.csv"
    if not overall_path.exists() or not by_horizon_path.exists() or not row_level_path.exists():
        raise FileNotFoundError(f"Missing one or more saved ablation outputs in {ablation_dir}")
    overall_df = pd.read_csv(overall_path)
    by_horizon_df = pd.read_csv(by_horizon_path)
    long_df = pd.read_csv(row_level_path)
    return overall_df, by_horizon_df, long_df


def has_existing_ablation_outputs(ablation_dir: Path) -> bool:
    return (ablation_dir / "overall_metrics.csv").exists() and \
           (ablation_dir / "metrics_by_horizon.csv").exists() and \
           (ablation_dir / "forecast_row_level_metrics.csv").exists()



# ============================================================
# Advanced diagnostics for model-structure optimization
# ============================================================
def ensure_long_df_time_columns(long_df: pd.DataFrame) -> pd.DataFrame:
    out = long_df.copy()
    if "timestamp_utc" in out.columns:
        out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
        out["timestamp_local"] = out["timestamp_utc"].dt.tz_convert("US/Eastern")
    elif "timestamp_local" in out.columns:
        out["timestamp_local"] = pd.to_datetime(out["timestamp_local"], errors="coerce")
        if getattr(out["timestamp_local"].dt, "tz", None) is None:
            out["timestamp_local"] = out["timestamp_local"].dt.tz_localize("US/Eastern", nonexistent="shift_forward", ambiguous="NaT")
        out["timestamp_utc"] = out["timestamp_local"].dt.tz_convert("UTC")
    if "timestamp_local" in out.columns:
        out["hour_local"] = out["timestamp_local"].dt.hour
        out["dow_local"] = out["timestamp_local"].dt.dayofweek
        out["month_local"] = out["timestamp_local"].dt.month
        out["year_local"] = out["timestamp_local"].dt.year
        out["is_weekend"] = (out["dow_local"] >= 5).astype(int)
    return out


def approx_wasserstein_1d(a: np.ndarray, b: np.ndarray, q_points: int = 199) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return math.nan
    qs = np.linspace(0.0, 1.0, q_points)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.mean(np.abs(qa - qb)))


def effective_rank_from_values(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 1e-12)]
    if vals.size == 0:
        return math.nan
    probs = vals / np.sum(vals)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(np.exp(entropy))


def compute_pca_summary(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    analysis_name: str,
    standardize: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix for PCA summary, got shape {x.shape}")
    valid_mask = np.isfinite(x).all(axis=1)
    x = x[valid_mask]
    if x.shape[0] < 2 or x.shape[1] < 1:
        return pd.DataFrame(), pd.DataFrame()

    x = x - x.mean(axis=0, keepdims=True)
    if standardize:
        scale = x.std(axis=0, keepdims=True)
        scale[scale < 1e-8] = 1.0
        x = x / scale

    _, s, vt = np.linalg.svd(x, full_matrices=False)
    denom = max(x.shape[0] - 1, 1)
    explained_var = (s ** 2) / float(denom)
    explained_ratio = explained_var / max(float(explained_var.sum()), 1e-12)

    explained_rows = []
    for i, ratio in enumerate(explained_ratio, start=1):
        explained_rows.append({
            "analysis": analysis_name,
            "component": i,
            "explained_variance": float(explained_var[i - 1]),
            "explained_ratio": float(ratio),
            "cumulative_explained_ratio": float(np.sum(explained_ratio[:i])),
        })

    loading_rows = []
    loadings = vt.T
    for fi, feature_name in enumerate(feature_names):
        row = {"analysis": analysis_name, "feature": str(feature_name)}
        for ci in range(loadings.shape[1]):
            row[f"pc{ci+1}"] = float(loadings[fi, ci])
        loading_rows.append(row)

    return pd.DataFrame(explained_rows), pd.DataFrame(loading_rows)


def deseasonalize_energy_by_calendar(energy_df_with_cal: pd.DataFrame, zone_cols: Sequence[str]) -> pd.DataFrame:
    season_keys = ["hour_local", "dow_local", "month", "is_weekend"]
    seasonal_mean = energy_df_with_cal.groupby(season_keys)[list(zone_cols)].transform("mean")
    return energy_df_with_cal[list(zone_cols)] - seasonal_mean


def plot_heatmap(
    matrix: np.ndarray,
    x_labels: Sequence[object],
    y_labels: Sequence[object],
    title: str,
    out_path: Path,
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.size == 0:
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(x_labels)), max(5, 0.45 * len(y_labels))))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(x) for x in x_labels], rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([str(y) for y in y_labels])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_scree(explained_df: pd.DataFrame, title: str, out_path: Path) -> None:
    if explained_df.empty:
        return
    sdf = explained_df.sort_values("component")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sdf["component"], sdf["explained_ratio"], marker="o", label="Explained ratio")
    ax.plot(sdf["component"], sdf["cumulative_explained_ratio"], marker="o", label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance ratio")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_sampled_weather_feature_frame(
    energy_df: pd.DataFrame,
    zone_cols: Sequence[str],
    weather_lookup: Dict[str, Path],
    weather_var_names: Sequence[str],
    weather_stride: int,
    max_weather_files: Optional[int],
    tile_grid: Tuple[int, int] = (6, 6),
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    sampled_count = 0
    grid_h, grid_w = int(tile_grid[0]), int(tile_grid[1])

    for i, ts in enumerate(tqdm(energy_df.index, desc="Building advanced weather feature frame")):
        if weather_stride > 1 and (i % weather_stride) != 0:
            continue
        key = ts.strftime("%Y%m%d%H")
        if key not in weather_lookup:
            continue
        x = safe_torch_load(weather_lookup[key])
        if x.ndim != 3:
            continue
        x_np = x.cpu().numpy()
        h, w, n_channels = x_np.shape
        row_splits = np.array_split(np.arange(h), grid_h)
        col_splits = np.array_split(np.arange(w), grid_w)

        rec: Dict[str, float] = {
            "timestamp": ts,
            "hour_local": float(ts.tz_convert("US/Eastern").hour),
            "month_local": float(ts.tz_convert("US/Eastern").month),
            "is_weekend": float(int(ts.tz_convert("US/Eastern").dayofweek >= 5)),
        }
        for zone in zone_cols:
            rec[f"target__{zone}"] = float(energy_df.iloc[i][zone])
        rec["target__total"] = float(energy_df.iloc[i][list(zone_cols)].sum())

        for c in range(n_channels):
            cname = weather_var_names[c] if c < len(weather_var_names) else f"ch_{c}"
            plane = x_np[:, :, c]
            rec[f"mean__{cname}"] = float(np.mean(plane))
            rec[f"std__{cname}"] = float(np.std(plane))
            rec[f"min__{cname}"] = float(np.min(plane))
            rec[f"max__{cname}"] = float(np.max(plane))
            for rr, ridx in enumerate(row_splits):
                for cc, cidx in enumerate(col_splits):
                    block = plane[np.ix_(ridx, cidx)]
                    rec[f"tilemean__{cname}__r{rr}_c{cc}"] = float(np.mean(block))
        records.append(rec)
        sampled_count += 1
        if max_weather_files is not None and sampled_count >= max_weather_files:
            break

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


def compute_year_distribution_shift(
    energy_only_df: pd.DataFrame,
    zone_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(pd.unique(energy_only_df.index.year))
    yearly_rows = []
    pair_rows = []

    for year in years:
        sdf = energy_only_df.loc[energy_only_df.index.year == year, list(zone_cols)]
        for zone in zone_cols:
            vals = pd.to_numeric(sdf[zone], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if vals.size == 0:
                continue
            yearly_rows.append({
                "year": int(year),
                "zone": zone,
                "count": int(vals.size),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "p05": float(np.quantile(vals, 0.05)),
                "p50": float(np.quantile(vals, 0.50)),
                "p95": float(np.quantile(vals, 0.95)),
            })

    stats_df = pd.DataFrame(yearly_rows)
    if stats_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    for zone in zone_cols:
        zone_series = {
            int(year): pd.to_numeric(
                energy_only_df.loc[energy_only_df.index.year == year, zone],
                errors="coerce",
            ).dropna().to_numpy(dtype=np.float64)
            for year in years
        }
        for i, year_a in enumerate(years):
            for year_b in years[i + 1:]:
                a = zone_series[int(year_a)]
                b = zone_series[int(year_b)]
                if a.size == 0 or b.size == 0:
                    continue
                mean_a = float(np.mean(a))
                mean_b = float(np.mean(b))
                std_a = float(np.std(a))
                std_b = float(np.std(b))
                pair_rows.append({
                    "zone": zone,
                    "year_a": int(year_a),
                    "year_b": int(year_b),
                    "mean_shift_pct": float(100.0 * (mean_b - mean_a) / max(abs(mean_a), 1e-6)),
                    "std_ratio": float(std_b / max(std_a, 1e-6)),
                    "median_shift_pct": float(100.0 * (np.median(b) - np.median(a)) / max(abs(np.median(a)), 1e-6)),
                    "approx_wasserstein": approx_wasserstein_1d(a, b),
                })
    return stats_df, pd.DataFrame(pair_rows)


def compute_baseline_by_year(
    energy_only_df: pd.DataFrame,
    zone_cols: Sequence[str],
    history_len: int,
    future_len: int,
) -> pd.DataFrame:
    rows = []
    for year in sorted(pd.unique(energy_only_df.index.year)):
        try:
            overall_df, _, _ = compute_horizon_baselines(energy_only_df, zone_cols, [int(year)], history_len, future_len)
        except Exception:
            continue
        for _, row in overall_df.iterrows():
            rows.append({
                "year": int(year),
                "baseline": str(row["baseline"]),
                "overall_mape": float(row["overall_mape"]),
                "num_windows": int(row["num_windows"]),
            })
    return pd.DataFrame(rows)


def compute_weather_channel_redundancy(sampled_weather_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sampled_weather_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    mean_cols = sorted([c for c in sampled_weather_df.columns if c.startswith("mean__")])
    if len(mean_cols) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    x = sampled_weather_df[mean_cols].to_numpy(dtype=np.float64)
    corr = pd.DataFrame(x, columns=mean_cols).corr()

    pair_rows = []
    for i, col_a in enumerate(mean_cols):
        for col_b in mean_cols[i + 1:]:
            pair_rows.append({
                "feature_a": col_a,
                "feature_b": col_b,
                "corr": float(corr.loc[col_a, col_b]),
                "abs_corr": float(abs(corr.loc[col_a, col_b])),
            })
    pair_df = pd.DataFrame(pair_rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)

    explained_df, _ = compute_pca_summary(x, mean_cols, analysis_name="weather_spatial_mean_features", standardize=True)
    summary_rows = []
    evals = np.linalg.eigvalsh(corr.to_numpy(dtype=np.float64))
    summary_rows.append({
        "num_features": int(len(mean_cols)),
        "effective_rank_corr": effective_rank_from_values(evals),
        "mean_abs_offdiag_corr": float(np.nanmean(np.abs(corr.to_numpy(dtype=np.float64)[~np.eye(len(mean_cols), dtype=bool)]))),
        "top_pc_cumulative_3": float(explained_df.loc[explained_df["component"] <= 3, "explained_ratio"].sum()) if not explained_df.empty else math.nan,
    })
    return pd.DataFrame(summary_rows), pair_df, corr


def parse_tile_feature_name(feature_name: str) -> Tuple[str, int, int]:
    rest = feature_name.split("tilemean__", 1)[1]
    channel, tail = rest.split("__r", 1)
    row_txt, col_txt = tail.split("_c", 1)
    return channel, int(row_txt), int(col_txt)


def compute_zone_specific_weather_relevance(
    sampled_weather_df: pd.DataFrame,
    zone_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if sampled_weather_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    tile_cols = [c for c in sampled_weather_df.columns if c.startswith("tilemean__")]
    if len(tile_cols) == 0:
        return pd.DataFrame(), pd.DataFrame()

    tile_rows = []
    summary_rows = []
    channels = sorted({parse_tile_feature_name(col)[0] for col in tile_cols})

    for zone in zone_cols:
        y = sampled_weather_df[f"target__{zone}"].to_numpy(dtype=np.float64)
        for channel in channels:
            channel_tile_rows = []
            for col in [c for c in tile_cols if c.startswith(f"tilemean__{channel}__")]:
                _, rr, cc = parse_tile_feature_name(col)
                x = sampled_weather_df[col].to_numpy(dtype=np.float64)
                mask = np.isfinite(x) & np.isfinite(y)
                corr_val = np.corrcoef(x[mask], y[mask])[0, 1] if int(mask.sum()) >= 3 else math.nan
                row = {
                    "zone": zone,
                    "channel": channel,
                    "tile_row": int(rr),
                    "tile_col": int(cc),
                    "corr": float(corr_val) if not math.isnan(corr_val) else math.nan,
                    "abs_corr": float(abs(corr_val)) if not math.isnan(corr_val) else math.nan,
                }
                tile_rows.append(row)
                channel_tile_rows.append(row)
            valid_abs = [r["abs_corr"] for r in channel_tile_rows if not math.isnan(r["abs_corr"])]
            if len(valid_abs) == 0:
                continue
            best_row = max(channel_tile_rows, key=lambda r: -1.0 if math.isnan(r["abs_corr"]) else r["abs_corr"])
            summary_rows.append({
                "zone": zone,
                "channel": channel,
                "max_abs_corr": float(np.max(valid_abs)),
                "mean_abs_corr": float(np.mean(valid_abs)),
                "best_tile_row": int(best_row["tile_row"]),
                "best_tile_col": int(best_row["tile_col"]),
                "best_tile_corr": float(best_row["corr"]),
            })
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["zone", "max_abs_corr"], ascending=[True, False]).reset_index(drop=True)
    return pd.DataFrame(tile_rows), summary_df


def plot_weather_relevance_heatmaps(
    tile_corr_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_dir: Path,
    top_k_channels: int = 3,
) -> None:
    ensure_dir(out_dir)
    if tile_corr_df.empty or summary_df.empty:
        return
    for zone, zdf in summary_df.groupby("zone"):
        top_channels = zdf.sort_values("max_abs_corr", ascending=False).head(top_k_channels)["channel"].tolist()
        for channel in top_channels:
            sdf = tile_corr_df[(tile_corr_df["zone"] == zone) & (tile_corr_df["channel"] == channel)].copy()
            if sdf.empty:
                continue
            pivot = sdf.pivot(index="tile_row", columns="tile_col", values="corr").sort_index().sort_index(axis=1)
            plot_heatmap(
                pivot.to_numpy(dtype=np.float64),
                list(pivot.columns),
                list(pivot.index),
                title=f"Weather relevance for {zone} | {channel}",
                out_path=out_dir / f"weather_relevance_{zone}_{channel}.png",
                xlabel="Tile column",
                ylabel="Tile row",
            )


def compute_skill_atlas(long_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    work = ensure_long_df_time_columns(long_df)
    work["skill_delta_vs_prev_day"] = work["baseline_abs_pct_error"] - work["abs_pct_error"]
    work["model_win_vs_prev_day"] = (work["abs_pct_error"] < work["baseline_abs_pct_error"]).astype(int)
    groupings = {
        "by_zone": ["zone"],
        "by_horizon": ["forecast_horizon"],
        "by_zone_horizon": ["zone", "forecast_horizon"],
        "by_hour": ["hour_local"],
        "by_month": ["month_local"],
        "by_weekend": ["is_weekend"],
        "by_target_quantile": ["target_quantile_bin"],
        "by_ramp_quantile": ["abs_ramp_quantile_bin"],
    }
    outputs: Dict[str, pd.DataFrame] = {}
    for name, cols in groupings.items():
        rows = []
        for key, sdf in work.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            row = {col: val for col, val in zip(cols, key)}
            row.update({
                "count": int(len(sdf)),
                "model_mape": float(sdf["abs_pct_error"].mean()),
                "prev_day_mape": float(sdf["baseline_abs_pct_error"].mean()),
                "skill_delta_vs_prev_day": float(sdf["skill_delta_vs_prev_day"].mean()),
                "median_skill_delta_vs_prev_day": float(sdf["skill_delta_vs_prev_day"].median()),
                "model_win_rate_pct": float(100.0 * sdf["model_win_vs_prev_day"].mean()),
            })
            rows.append(row)
        outputs[name] = pd.DataFrame(rows)
    outputs["row_level"] = work
    return outputs


def plot_skill_atlas(skill_outputs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    ensure_dir(out_dir)
    by_zone_horizon = skill_outputs.get("by_zone_horizon", pd.DataFrame())
    if not by_zone_horizon.empty:
        pivot = by_zone_horizon.pivot(index="zone", columns="forecast_horizon", values="skill_delta_vs_prev_day")
        plot_heatmap(
            pivot.to_numpy(dtype=np.float64),
            list(pivot.columns),
            list(pivot.index),
            title="Skill delta vs prev-day by zone and horizon",
            out_path=out_dir / "skill_delta_zone_horizon_heatmap.png",
            xlabel="Forecast horizon",
            ylabel="Zone",
        )

    by_zone = skill_outputs.get("by_zone", pd.DataFrame())
    if not by_zone.empty:
        sdf = by_zone.sort_values("skill_delta_vs_prev_day", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(sdf["zone"], sdf["skill_delta_vs_prev_day"])
        ax.set_ylabel("Prev-day MAPE minus model MAPE")
        ax.set_title("Average model uplift vs prev-day by zone")
        fig.tight_layout()
        fig.savefig(out_dir / "skill_delta_by_zone.png", dpi=160)
        plt.close(fig)

    by_hour = skill_outputs.get("by_hour", pd.DataFrame())
    if not by_hour.empty:
        sdf = by_hour.sort_values("hour_local")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sdf["hour_local"], sdf["skill_delta_vs_prev_day"], marker="o")
        ax.set_xlabel("Local hour")
        ax.set_ylabel("Prev-day MAPE minus model MAPE")
        ax.set_title("Average model uplift vs prev-day by local hour")
        fig.tight_layout()
        fig.savefig(out_dir / "skill_delta_by_hour.png", dpi=160)
        plt.close(fig)


def build_hourly_residual_frame(long_df: pd.DataFrame) -> pd.DataFrame:
    work = ensure_long_df_time_columns(long_df)
    work["baseline_error"] = work["prev_day_baseline"] - work["target"]
    grouped = work.groupby(["timestamp_utc", "zone"], as_index=False).agg({
        "error": "mean",
        "baseline_error": "mean",
        "abs_error": "mean",
        "abs_pct_error": "mean",
        "baseline_abs_pct_error": "mean",
        "target": "mean",
    })
    grouped = grouped.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)
    return grouped


def compute_residual_memory_analysis(hourly_residual_df: pd.DataFrame, zone_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if hourly_residual_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    acf_rows = []
    summary_rows = []

    for zone in zone_cols:
        sdf = hourly_residual_df[hourly_residual_df["zone"] == zone].sort_values("timestamp_utc")
        for series_name, col in [("model_residual", "error"), ("prev_day_residual", "baseline_error")]:
            s = pd.Series(sdf[col].to_numpy(dtype=np.float64))
            ac_vals = []
            for lag in lags:
                ac = float(s.autocorr(lag=lag)) if len(s) > lag + 2 else math.nan
                acf_rows.append({
                    "zone": zone,
                    "series": series_name,
                    "lag_hours": int(lag),
                    "autocorr": ac,
                })
                if not math.isnan(ac):
                    ac_vals.append((lag, ac))
            if len(ac_vals) > 0:
                n = float(len(s))
                q_stat = float(sum((n * (n + 2.0) * (ac ** 2)) / max(n - lag, 1.0) for lag, ac in ac_vals))
            else:
                q_stat = math.nan
            summary_rows.append({
                "zone": zone,
                "series": series_name,
                "num_points": int(len(s)),
                "mean_abs_autocorr": float(np.nanmean([abs(ac) for _, ac in ac_vals])) if len(ac_vals) > 0 else math.nan,
                "ljung_box_style_q": q_stat,
            })
    return pd.DataFrame(acf_rows), pd.DataFrame(summary_rows)


def plot_residual_memory(acf_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    if acf_df.empty:
        return
    for zone, zdf in acf_df.groupby("zone"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for series_name in zdf["series"].unique():
            sdf = zdf[zdf["series"] == series_name].sort_values("lag_hours")
            ax.plot(sdf["lag_hours"], sdf["autocorr"], marker="o", label=series_name)
        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("Residual autocorrelation")
        ax.set_title(f"Residual memory by lag | {zone}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"residual_memory_{zone}.png", dpi=160)
        plt.close(fig)

    pivot_rows = []
    for zone in sorted(acf_df["zone"].unique()):
        for lag in sorted(acf_df["lag_hours"].unique()):
            model_row = acf_df[(acf_df["zone"] == zone) & (acf_df["lag_hours"] == lag) & (acf_df["series"] == "model_residual")]
            base_row = acf_df[(acf_df["zone"] == zone) & (acf_df["lag_hours"] == lag) & (acf_df["series"] == "prev_day_residual")]
            if len(model_row) and len(base_row):
                pivot_rows.append({
                    "zone": zone,
                    "lag_hours": lag,
                    "delta_abs_autocorr": abs(float(base_row["autocorr"].iloc[0])) - abs(float(model_row["autocorr"].iloc[0])),
                })
    pivot_df = pd.DataFrame(pivot_rows)
    if not pivot_df.empty:
        pivot = pivot_df.pivot(index="zone", columns="lag_hours", values="delta_abs_autocorr")
        plot_heatmap(
            pivot.to_numpy(dtype=np.float64),
            list(pivot.columns),
            list(pivot.index),
            title="Reduction in |residual autocorr| vs prev-day",
            out_path=out_dir / "residual_memory_delta_heatmap.png",
            xlabel="Lag (hours)",
            ylabel="Zone",
        )


def compute_conditional_error_volatility(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = ensure_long_df_time_columns(long_df)
    work["baseline_error"] = work["prev_day_baseline"] - work["target"]

    zone_h_rows = []
    for key, sdf in work.groupby(["zone", "forecast_horizon"], dropna=False):
        zone, horizon = key
        target_std = float(np.std(sdf["target"].to_numpy(dtype=np.float64)))
        row = {
            "zone": zone,
            "forecast_horizon": int(horizon),
            "count": int(len(sdf)),
            "target_std": target_std,
            "model_error_std": float(np.std(sdf["error"].to_numpy(dtype=np.float64))),
            "baseline_error_std": float(np.std(sdf["baseline_error"].to_numpy(dtype=np.float64))),
            "model_abs_error_mean": float(np.mean(sdf["abs_error"].to_numpy(dtype=np.float64))),
            "baseline_abs_error_mean": float(np.mean(np.abs(sdf["baseline_error"].to_numpy(dtype=np.float64)))),
            "model_abs_pct_error_mean": float(np.mean(sdf["abs_pct_error"].to_numpy(dtype=np.float64))),
            "baseline_abs_pct_error_mean": float(np.mean(sdf["baseline_abs_pct_error"].to_numpy(dtype=np.float64))),
            "model_abs_error_over_target_std": float(np.mean(sdf["abs_error"].to_numpy(dtype=np.float64)) / max(target_std, 1e-6)),
        }
        zone_h_rows.append(row)

    month_hour_rows = []
    for key, sdf in work.groupby(["month_local", "hour_local"], dropna=False):
        month_local, hour_local = key
        month_hour_rows.append({
            "month_local": int(month_local),
            "hour_local": int(hour_local),
            "count": int(len(sdf)),
            "model_abs_pct_error_mean": float(np.mean(sdf["abs_pct_error"].to_numpy(dtype=np.float64))),
            "baseline_abs_pct_error_mean": float(np.mean(sdf["baseline_abs_pct_error"].to_numpy(dtype=np.float64))),
            "model_error_std": float(np.std(sdf["error"].to_numpy(dtype=np.float64))),
            "baseline_error_std": float(np.std((sdf["prev_day_baseline"] - sdf["target"]).to_numpy(dtype=np.float64))),
        })

    weekend_rows = []
    for key, sdf in work.groupby(["zone", "is_weekend"], dropna=False):
        zone, is_weekend = key
        weekend_rows.append({
            "zone": zone,
            "is_weekend": int(is_weekend),
            "count": int(len(sdf)),
            "model_abs_pct_error_mean": float(np.mean(sdf["abs_pct_error"].to_numpy(dtype=np.float64))),
            "baseline_abs_pct_error_mean": float(np.mean(sdf["baseline_abs_pct_error"].to_numpy(dtype=np.float64))),
            "model_error_std": float(np.std(sdf["error"].to_numpy(dtype=np.float64))),
            "baseline_error_std": float(np.std((sdf["prev_day_baseline"] - sdf["target"]).to_numpy(dtype=np.float64))),
        })

    return pd.DataFrame(zone_h_rows), pd.DataFrame(month_hour_rows), pd.DataFrame(weekend_rows)


def plot_conditional_error_volatility(zone_h_df: pd.DataFrame, month_hour_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    if not zone_h_df.empty:
        pivot = zone_h_df.pivot(index="zone", columns="forecast_horizon", values="model_error_std")
        plot_heatmap(
            pivot.to_numpy(dtype=np.float64),
            list(pivot.columns),
            list(pivot.index),
            title="Model residual std by zone and horizon",
            out_path=out_dir / "model_residual_std_zone_horizon.png",
            xlabel="Forecast horizon",
            ylabel="Zone",
        )

    if not month_hour_df.empty:
        pivot = month_hour_df.pivot(index="month_local", columns="hour_local", values="model_abs_pct_error_mean")
        plot_heatmap(
            pivot.to_numpy(dtype=np.float64),
            list(pivot.columns),
            list(pivot.index),
            title="Model MAPE by month and local hour",
            out_path=out_dir / "model_mape_month_hour.png",
            xlabel="Local hour",
            ylabel="Month",
        )


def compute_extreme_event_cohorts(long_df: pd.DataFrame) -> pd.DataFrame:
    work = ensure_long_df_time_columns(long_df)
    work = work.copy()
    work["baseline_error"] = work["prev_day_baseline"] - work["target"]
    zone_q95 = work.groupby("zone")["target"].transform(lambda s: s.quantile(0.95))
    zone_q05 = work.groupby("zone")["target"].transform(lambda s: s.quantile(0.05))
    ramp_q95 = work.groupby("zone")["abs_ramp_1h"].transform(lambda s: s.quantile(0.95))

    cohort_masks = {
        "all_rows": np.ones(len(work), dtype=bool),
        "top_5pct_load_per_zone": work["target"] >= zone_q95,
        "bottom_5pct_load_per_zone": work["target"] <= zone_q05,
        "top_5pct_abs_ramp_per_zone": work["abs_ramp_1h"] >= ramp_q95,
        "weekends": work["is_weekend"] == 1,
        "weekday_business_hours": (work["is_weekend"] == 0) & (work["hour_local"].between(8, 17)),
        "evening_peak_hours": work["hour_local"].between(16, 20),
        "overnight_hours": work["hour_local"].between(0, 5),
        "winter_months": work["month_local"].isin([12, 1, 2]),
        "summer_months": work["month_local"].isin([6, 7, 8]),
    }

    rows = []
    total = max(len(work), 1)
    for cohort_name, mask in cohort_masks.items():
        sdf = work.loc[np.asarray(mask)]
        if sdf.empty:
            continue
        model_mape = float(np.mean(sdf["abs_pct_error"].to_numpy(dtype=np.float64)))
        baseline_mape = float(np.mean(sdf["baseline_abs_pct_error"].to_numpy(dtype=np.float64)))
        rows.append({
            "cohort": cohort_name,
            "count": int(len(sdf)),
            "share_pct": float(100.0 * len(sdf) / total),
            "model_mape": model_mape,
            "prev_day_mape": baseline_mape,
            "skill_delta_vs_prev_day": baseline_mape - model_mape,
        })
    return pd.DataFrame(rows).sort_values("skill_delta_vs_prev_day", ascending=False).reset_index(drop=True)


def plot_extreme_event_cohorts(cohort_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    if cohort_df.empty:
        return
    sdf = cohort_df.sort_values("skill_delta_vs_prev_day", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(sdf["cohort"], sdf["skill_delta_vs_prev_day"])
    ax.set_ylabel("Prev-day MAPE minus model MAPE")
    ax.set_title("Model uplift vs prev-day across extreme-event cohorts")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "extreme_event_cohort_skill.png", dpi=160)
    plt.close(fig)


def compute_prediction_calibration(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = ensure_long_df_time_columns(long_df)
    work = work.copy()
    denom = np.maximum(np.abs(work["target"].to_numpy(dtype=np.float64)), 1e-6)
    work["pred_over_target"] = work["pred"].to_numpy(dtype=np.float64) / denom
    work["baseline_over_target"] = work["prev_day_baseline"].to_numpy(dtype=np.float64) / denom

    if "year_local" not in work.columns:
        work["year_local"] = work["timestamp_local"].dt.year
    work["target_decile"] = pd.qcut(
        work["target"].rank(method="first"),
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
    )

    by_year = []
    for year, sdf in work.groupby("year_local"):
        by_year.append({
            "year_local": int(year),
            "count": int(len(sdf)),
            "mean_pred_over_target": float(np.mean(sdf["pred_over_target"].to_numpy(dtype=np.float64))),
            "mean_baseline_over_target": float(np.mean(sdf["baseline_over_target"].to_numpy(dtype=np.float64))),
            "mean_pred": float(np.mean(sdf["pred"].to_numpy(dtype=np.float64))),
            "mean_target": float(np.mean(sdf["target"].to_numpy(dtype=np.float64))),
        })

    by_month = []
    for month, sdf in work.groupby("month_local"):
        by_month.append({
            "month_local": int(month),
            "count": int(len(sdf)),
            "mean_pred_over_target": float(np.mean(sdf["pred_over_target"].to_numpy(dtype=np.float64))),
            "mean_baseline_over_target": float(np.mean(sdf["baseline_over_target"].to_numpy(dtype=np.float64))),
        })

    by_decile = []
    for decile, sdf in work.groupby("target_decile", dropna=False):
        by_decile.append({
            "target_decile": str(decile),
            "count": int(len(sdf)),
            "mean_pred": float(np.mean(sdf["pred"].to_numpy(dtype=np.float64))),
            "mean_target": float(np.mean(sdf["target"].to_numpy(dtype=np.float64))),
            "mean_pred_over_target": float(np.mean(sdf["pred_over_target"].to_numpy(dtype=np.float64))),
            "mean_baseline_over_target": float(np.mean(sdf["baseline_over_target"].to_numpy(dtype=np.float64))),
        })

    return pd.DataFrame(by_year), pd.DataFrame(by_month), pd.DataFrame(by_decile)


def plot_prediction_calibration(by_year_df: pd.DataFrame, by_month_df: pd.DataFrame, by_decile_df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    if not by_year_df.empty and len(by_year_df) > 1:
        sdf = by_year_df.sort_values("year_local")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sdf["year_local"], sdf["mean_pred_over_target"], marker="o", label="model")
        ax.plot(sdf["year_local"], sdf["mean_baseline_over_target"], marker="o", label="prev_day")
        ax.axhline(1.0, linestyle="--")
        ax.set_xlabel("Local year")
        ax.set_ylabel("Mean prediction / target")
        ax.set_title("Calibration drift by year")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "calibration_by_year.png", dpi=160)
        plt.close(fig)

    if not by_month_df.empty:
        sdf = by_month_df.sort_values("month_local")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sdf["month_local"], sdf["mean_pred_over_target"], marker="o", label="model")
        ax.plot(sdf["month_local"], sdf["mean_baseline_over_target"], marker="o", label="prev_day")
        ax.axhline(1.0, linestyle="--")
        ax.set_xlabel("Month")
        ax.set_ylabel("Mean prediction / target")
        ax.set_title("Calibration by month")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "calibration_by_month.png", dpi=160)
        plt.close(fig)

    if not by_decile_df.empty:
        sdf = by_decile_df.copy()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(sdf["target_decile"], sdf["mean_target"], marker="o", label="mean target")
        ax.plot(sdf["target_decile"], sdf["mean_pred"], marker="o", label="mean pred")
        ax.set_xlabel("Target decile")
        ax.set_ylabel("Load")
        ax.set_title("Calibration by target decile")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "calibration_by_target_decile.png", dpi=160)
        plt.close(fig)


def write_advanced_summary(
    out_path: Path,
    year_pair_df: pd.DataFrame,
    shared_factor_raw_df: pd.DataFrame,
    weather_relevance_summary_df: pd.DataFrame,
    skill_zone_h_df: pd.DataFrame,
    residual_memory_summary_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    weather_redundancy_summary_df: pd.DataFrame,
    calibration_year_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("Advanced diagnostics summary")
    lines.append("=" * 32)
    lines.append("")

    if not year_pair_df.empty:
        top_shift = year_pair_df.sort_values("approx_wasserstein", ascending=False).iloc[0]
        lines.append(
            f"Largest year-to-year distribution shift: zone={top_shift['zone']}, years={int(top_shift['year_a'])}->{int(top_shift['year_b'])}, "
            f"approx_wasserstein={top_shift['approx_wasserstein']:.4f}, mean_shift_pct={top_shift['mean_shift_pct']:.2f}%"
        )
    if not shared_factor_raw_df.empty:
        first_pc = shared_factor_raw_df.loc[shared_factor_raw_df["component"] == 1, "explained_ratio"]
        if len(first_pc):
            lines.append(f"Raw demand first principal component variance share: {float(first_pc.iloc[0]):.4f}")
    if not weather_relevance_summary_df.empty:
        top_weather = weather_relevance_summary_df.sort_values("max_abs_corr", ascending=False).iloc[0]
        lines.append(
            f"Strongest zone-specific weather map signal: zone={top_weather['zone']}, channel={top_weather['channel']}, "
            f"max_abs_corr={top_weather['max_abs_corr']:.4f} at tile ({int(top_weather['best_tile_row'])}, {int(top_weather['best_tile_col'])})"
        )
    if not skill_zone_h_df.empty:
        best_slice = skill_zone_h_df.sort_values("skill_delta_vs_prev_day", ascending=False).iloc[0]
        worst_slice = skill_zone_h_df.sort_values("skill_delta_vs_prev_day", ascending=True).iloc[0]
        lines.append(
            f"Best model-vs-baseline slice: zone={best_slice['zone']}, horizon={int(best_slice['forecast_horizon'])}, "
            f"skill_delta={best_slice['skill_delta_vs_prev_day']:.4f} MAPE points"
        )
        lines.append(
            f"Worst model-vs-baseline slice: zone={worst_slice['zone']}, horizon={int(worst_slice['forecast_horizon'])}, "
            f"skill_delta={worst_slice['skill_delta_vs_prev_day']:.4f} MAPE points"
        )
    if not residual_memory_summary_df.empty:
        merged = residual_memory_summary_df.pivot(index="zone", columns="series", values="mean_abs_autocorr")
        if {"model_residual", "prev_day_residual"}.issubset(set(merged.columns)):
            merged = merged.assign(delta=merged["prev_day_residual"] - merged["model_residual"])
            best_zone = merged["delta"].sort_values(ascending=False).index[0]
            lines.append(
                f"Largest reduction in mean absolute residual autocorr vs prev-day: zone={best_zone}, delta={float(merged.loc[best_zone, 'delta']):.4f}"
            )
    if not cohort_df.empty:
        best_cohort = cohort_df.sort_values("skill_delta_vs_prev_day", ascending=False).iloc[0]
        lines.append(
            f"Best extreme-event cohort for model uplift: cohort={best_cohort['cohort']}, "
            f"skill_delta={best_cohort['skill_delta_vs_prev_day']:.4f} MAPE points"
        )
    if not weather_redundancy_summary_df.empty:
        row = weather_redundancy_summary_df.iloc[0]
        lines.append(
            f"Weather feature redundancy summary: effective_rank_corr={row['effective_rank_corr']:.4f}, "
            f"mean_abs_offdiag_corr={row['mean_abs_offdiag_corr']:.4f}, top_pc_cumulative_3={row['top_pc_cumulative_3']:.4f}"
        )
    if not calibration_year_df.empty and len(calibration_year_df) > 1:
        year_row = calibration_year_df.iloc[(calibration_year_df["mean_pred_over_target"] - 1.0).abs().argmax()]
        lines.append(
            f"Largest yearly calibration drift: year={int(year_row['year_local'])}, mean_pred_over_target={year_row['mean_pred_over_target']:.4f}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Extended dataset + inference-only analysis for energy/weather forecasting.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where all outputs will be written.")
    parser.add_argument("--weather-dir", type=str, default=WEATHER_DIR)
    parser.add_argument("--energy-dir", type=str, default=ENERGY_DIR)
    parser.add_argument("--train-years", type=str, default=",".join(map(str, TRAIN_YEARS)))
    parser.add_argument("--val-years", type=str, default=",".join(map(str, VAL_YEARS)))
    parser.add_argument("--test-years", type=str, default=",".join(map(str, TEST_YEARS)))
    parser.add_argument("--history-len", type=int, default=168, help="Raw history length used by evaluate.py-style windows.")
    parser.add_argument("--future-len", type=int, default=24)
    parser.add_argument("--max-acf-lag", type=int, default=336)
    parser.add_argument("--weather-stride", type=int, default=6)
    parser.add_argument("--max-weather-files", type=int, default=None)

    # Model / inference-only analysis
    parser.add_argument("--run-inference-analysis", action="store_true", help="Run inference-only analysis on an existing trained checkpoint.")
    parser.add_argument("--model-path", type=str, default="model.py", help="Path to model.py exposing get_model(metadata).")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Optional checkpoint override path.")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    parser.add_argument("--eval-years", type=str, default="", help="Years for inference analysis. Defaults to val-years if provided, else test-years.")
    parser.add_argument("--midnight-only", action="store_true", default=True, help="Use only midnight forecast starts, matching evaluate.py.")
    parser.add_argument("--all-hours", action="store_true", help="Use every hour as a forecast start instead of only midnight starts.")
    parser.add_argument("--max-eval-windows", type=int, default=None, help="Optional cap on number of eval windows.")
    parser.add_argument("--history-keep-hours", type=str, default="12,6,3,1", help="Comma-separated effective-history ablations within model-used history.")
    parser.add_argument("--run-channel-ablation", action="store_true", help="Zero each weather channel one at a time.")
    parser.add_argument("--blend-grid-step", type=float, default=0.05, help="Grid step for no-training blend alpha search.")
    parser.add_argument("--calibration-fraction", type=float, default=0.5, help="Fraction of eval windows used to fit blend weights.")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true", help="Reuse existing ablation outputs under output-dir/inference_only_analysis and continue from the first missing ablation.")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    csv_dir = out_dir / "csv"
    plot_dir = out_dir / "plots"
    text_dir = out_dir / "text"
    advanced_root = out_dir / "advanced_diagnostics"
    advanced_csv = advanced_root / "csv"
    advanced_plots = advanced_root / "plots"
    advanced_text = advanced_root / "text"
    ensure_dir(out_dir)
    ensure_dir(csv_dir)
    ensure_dir(plot_dir)
    ensure_dir(text_dir)
    ensure_dir(advanced_root)
    ensure_dir(advanced_csv)
    ensure_dir(advanced_plots)
    ensure_dir(advanced_text)

    train_years = parse_years(args.train_years)
    val_years = parse_years(args.val_years)
    test_years = parse_years(args.test_years)
    all_years = sorted(set(train_years) | set(val_years) | set(test_years))

    weather_dir = Path(args.weather_dir)
    energy_dir = Path(args.energy_dir)

    config = {
        "output_dir": str(out_dir),
        "weather_dir": str(weather_dir),
        "energy_dir": str(energy_dir),
        "train_years": train_years,
        "val_years": val_years,
        "test_years": test_years,
        "history_len": args.history_len,
        "future_len": args.future_len,
        "max_acf_lag": args.max_acf_lag,
        "weather_stride": args.weather_stride,
        "max_weather_files": args.max_weather_files,
        "run_inference_analysis": bool(args.run_inference_analysis),
        "model_path": args.model_path,
        "checkpoint_path": args.checkpoint_path,
        "device": args.device,
        "eval_years": args.eval_years,
        "midnight_only": bool(not args.all_hours),
        "max_eval_windows": args.max_eval_windows,
        "history_keep_hours": args.history_keep_hours,
        "run_channel_ablation": bool(args.run_channel_ablation),
        "blend_grid_step": args.blend_grid_step,
        "calibration_fraction": args.calibration_fraction,
        "advanced_diagnostics_dir": str(advanced_root),
        "advanced_weather_tile_grid": [6, 6],
        "advanced_weather_relevance_top_k": 3,
    }
    save_json(config, out_dir / "run_config.json")

    print("Loading energy data...")
    energy_df = load_energy_targets(energy_dir, all_years)
    zone_cols = list(energy_df.columns)
    energy_df_with_cal = add_calendar_columns(energy_df)
    energy_only_df = energy_df_with_cal[zone_cols].copy()

    print("Building weather file lookup...")
    weather_lookup = build_weather_lookup(weather_dir, all_years)
    weather_var_names = maybe_load_metadata(weather_dir)

    # --------------------------------------------------------
    # Core dataset summaries
    # --------------------------------------------------------
    contiguity_info = compute_contiguity_info(energy_only_df.index)
    save_json(contiguity_info, out_dir / "contiguity_summary.json")

    split_summary_df = compute_split_summary(energy_only_df, train_years, val_years, test_years)
    split_summary_df.to_csv(csv_dir / "split_summary.csv", index=False)

    window_summary_df, valid_windows_df = compute_window_validity(
        energy_only_df,
        weather_lookup,
        hist_len=args.history_len,
        fut_len=args.future_len,
    )
    window_summary_df.to_csv(csv_dir / "window_validity_summary.csv", index=False)
    valid_windows_df.to_csv(csv_dir / "valid_windows.csv", index=False)

    zone_stats_df = compute_zone_stats(energy_only_df, zone_cols)
    zone_stats_df.to_csv(csv_dir / "zone_stats.csv", index=False)

    regime_effect_df = compute_regime_effect_summary(energy_df_with_cal, zone_cols)
    regime_effect_df.to_csv(csv_dir / "regime_effect_summary.csv", index=False)

    hourly_profile_df = compute_group_profiles(energy_df_with_cal, zone_cols, "hour_local")
    hourly_profile_df.to_csv(csv_dir / "profile_by_hour_local.csv", index=False)

    dow_profile_df = compute_group_profiles(energy_df_with_cal, zone_cols, "dow_local")
    dow_profile_df.to_csv(csv_dir / "profile_by_dow_local.csv", index=False)

    month_profile_df = compute_group_profiles(energy_df_with_cal, zone_cols, "month")
    month_profile_df.to_csv(csv_dir / "profile_by_month.csv", index=False)

    weekend_profile_df = compute_group_profiles(energy_df_with_cal, zone_cols, "is_weekend")
    weekend_profile_df.to_csv(csv_dir / "profile_by_weekend.csv", index=False)

    holiday_profile_df = compute_group_profiles(energy_df_with_cal, zone_cols, "is_holiday")
    holiday_profile_df.to_csv(csv_dir / "profile_by_holiday.csv", index=False)

    zone_corr_df = energy_only_df.corr(numeric_only=True)
    zone_corr_df.to_csv(csv_dir / "zone_correlation.csv")

    acf_full_df, acf_selected_df = compute_autocorrelation_table(
        energy_only_df,
        zone_cols,
        max_lag=args.max_acf_lag,
        selected_lags=[1, 2, 3, 6, 12, 24, 48, 72, 168, 336],
    )
    acf_full_df.to_csv(csv_dir / "zone_autocorrelation_full.csv", index=False)
    acf_selected_df.to_csv(csv_dir / "zone_autocorrelation_selected.csv", index=False)

    ramp_stats_df = compute_ramp_stats(energy_only_df, zone_cols)
    ramp_stats_df.to_csv(csv_dir / "ramp_stats.csv", index=False)

    # --------------------------------------------------------
    # Baseline difficulty analysis
    # --------------------------------------------------------
    if len(val_years) > 0:
        val_baseline_overall_df, val_baseline_zone_df, val_baseline_horizon_df = compute_horizon_baselines(
            energy_only_df, zone_cols, val_years, args.history_len, args.future_len
        )
        val_baseline_overall_df.to_csv(csv_dir / "val_baseline_overall.csv", index=False)
        val_baseline_zone_df.to_csv(csv_dir / "val_baseline_by_zone.csv", index=False)
        val_baseline_horizon_df.to_csv(csv_dir / "val_baseline_by_horizon.csv", index=False)
    else:
        val_baseline_overall_df = pd.DataFrame()
        val_baseline_zone_df = pd.DataFrame()
        val_baseline_horizon_df = pd.DataFrame()

    if len(test_years) > 0:
        try:
            test_baseline_overall_df, test_baseline_zone_df, test_baseline_horizon_df = compute_horizon_baselines(
                energy_only_df, zone_cols, test_years, args.history_len, args.future_len
            )
        except Exception:
            test_baseline_overall_df = pd.DataFrame()
            test_baseline_zone_df = pd.DataFrame()
            test_baseline_horizon_df = pd.DataFrame()
        test_baseline_overall_df.to_csv(csv_dir / "test_baseline_overall.csv", index=False)
        test_baseline_zone_df.to_csv(csv_dir / "test_baseline_by_zone.csv", index=False)
        test_baseline_horizon_df.to_csv(csv_dir / "test_baseline_by_horizon.csv", index=False)
    else:
        test_baseline_overall_df = pd.DataFrame()
        test_baseline_zone_df = pd.DataFrame()
        test_baseline_horizon_df = pd.DataFrame()

    # --------------------------------------------------------
    # Weather analysis (sampled)
    # --------------------------------------------------------
    weather_stats_df, weather_corr_df = sample_weather_timeseries(
        energy_only_df,
        zone_cols,
        weather_lookup,
        weather_var_names,
        weather_stride=max(1, args.weather_stride),
        max_weather_files=args.max_weather_files,
    )
    weather_stats_df.to_csv(csv_dir / "weather_channel_stats_sampled.csv", index=False)
    weather_corr_df.to_csv(csv_dir / "weather_channel_demand_correlations_sampled.csv", index=False)

    # --------------------------------------------------------
    # Dataset plots
    # --------------------------------------------------------
    plot_zone_correlation(zone_corr_df, plot_dir / "zone_correlation_heatmap.png")
    plot_hourly_profiles(hourly_profile_df, zone_cols, plot_dir / "hourly_profile_by_zone.png")
    plot_acf_selected(acf_selected_df, plot_dir / "zone_autocorrelation_selected.png")
    if not val_baseline_horizon_df.empty:
        plot_baseline_horizon(val_baseline_horizon_df, plot_dir / "val_baseline_mape_by_horizon.png")
    if not test_baseline_horizon_df.empty:
        plot_baseline_horizon(test_baseline_horizon_df, plot_dir / "test_baseline_mape_by_horizon.png")

    # --------------------------------------------------------
    # Dataset recommendations
    # --------------------------------------------------------
    baseline_for_recs = test_baseline_overall_df if not test_baseline_overall_df.empty else val_baseline_overall_df
    recommendations = build_recommendations(
        zone_stats_df=zone_stats_df,
        regime_effect_df=regime_effect_df,
        acf_selected_df=acf_selected_df,
        zone_corr_df=zone_corr_df,
        baseline_overall_df=baseline_for_recs,
        weather_corr_df=weather_corr_df,
    )
    with open(text_dir / "architecture_recommendations.txt", "w", encoding="utf-8") as f:
        f.write(recommendations)

    quick_summary = {
        "num_zones": int(len(zone_cols)),
        "zone_names": list(zone_cols),
        "num_weather_files_indexed": int(len(weather_lookup)),
        "contiguity": contiguity_info,
        "outputs_written_to": str(out_dir),
    }
    save_json(quick_summary, out_dir / "quick_summary.json")

    # --------------------------------------------------------
    # Inference-only analysis
    # --------------------------------------------------------
    main_none_long_df = None
    eval_years_for_advanced: List[int] = []

    if args.run_inference_analysis:
        inference_root = out_dir / "inference_only_analysis"
        inference_csv = inference_root / "csv"
        inference_plots = inference_root / "plots"
        inference_text = inference_root / "text"
        ensure_dir(inference_root)
        ensure_dir(inference_csv)
        ensure_dir(inference_plots)
        ensure_dir(inference_text)

        eval_years = parse_years(args.eval_years)
        if len(eval_years) == 0:
            if len(val_years) > 0:
                eval_years = list(val_years)
            elif len(test_years) > 0:
                eval_years = list(test_years)
            else:
                raise ValueError("No eval years available. Provide --eval-years.")
        eval_years_for_advanced = list(eval_years)
        n_weather_vars = infer_num_weather_vars(weather_lookup)

        metadata = {
            "zone_names": zone_cols,
            "n_zones": len(zone_cols),
            "history_len": args.history_len,
            "future_len": args.future_len,
            "n_weather_vars": n_weather_vars,
        }

        device = choose_device(args.device)
        model_path = Path(args.model_path)
        checkpoint_override = Path(args.checkpoint_path) if args.checkpoint_path else None
        print(f"Loading model from: {model_path}")
        model = load_model_from_path(
            model_path=model_path,
            metadata=metadata,
            device=device,
            checkpoint_override=checkpoint_override,
        )

        ablations = make_ablation_list(
            model=model,
            run_channel_ablation=bool(args.run_channel_ablation),
            history_keep_hours=parse_int_list(args.history_keep_hours),
            n_weather_vars=n_weather_vars,
        )
        save_json({"ablations": ablations}, inference_root / "ablation_list.json")

        ablation_summary_rows = []
        all_ablation_horizon_rows = []
        main_none_long_df = None
        main_none_payload = None

        for ablation_name in ablations:
            ablation_dir = inference_root / "ablations" / ablation_name

            if args.resume and has_existing_ablation_outputs(ablation_dir):
                print(f"[resume] Reusing saved outputs for ablation: {ablation_name}")
                overall_df, by_horizon_df, long_df = read_existing_ablation_outputs(ablation_dir)
                payload = None
            else:
                payload = run_model_eval(
                    model=model,
                    energy_df=energy_only_df,
                    weather_lookup=weather_lookup,
                    zone_cols=zone_cols,
                    eval_years=eval_years,
                    raw_history_len=args.history_len,
                    future_len=args.future_len,
                    ablation_name=ablation_name,
                    device=device,
                    max_windows=args.max_eval_windows,
                    midnight_only=not args.all_hours,
                    log_every=args.log_every,
                )

                long_df = save_payload_metrics(payload, energy_only_df, ablation_dir)
                overall_df = pd.read_csv(ablation_dir / "overall_metrics.csv")
                by_horizon_df = pd.read_csv(ablation_dir / "metrics_by_horizon.csv")

            overall_mape = float(overall_df["mape"].iloc[0])
            prev_day_mape = float(overall_df["baseline_prev_day_mape"].iloc[0])
            ablation_summary_rows.append({
                "ablation": ablation_name,
                "overall_mape": overall_mape,
                "prev_day_baseline_mape": prev_day_mape,
                "delta_vs_none": math.nan,
                "delta_vs_prev_day": overall_mape - prev_day_mape,
            })

            tmp = by_horizon_df.copy()
            tmp["ablation"] = ablation_name
            all_ablation_horizon_rows.append(tmp)

            if ablation_name == "none":
                main_none_long_df = long_df.copy()
                main_none_payload = payload
                plot_model_vs_baseline_horizon(
                    long_df,
                    title=f"Model vs prev-day baseline by horizon ({','.join(map(str, eval_years))})",
                    out_path=inference_plots / "model_vs_prev_day_by_horizon.png",
                )
                plot_model_vs_baseline_zone(
                    long_df,
                    title=f"Model vs prev-day baseline by zone ({','.join(map(str, eval_years))})",
                    out_path=inference_plots / "model_vs_prev_day_by_zone.png",
                )

        ablation_summary_df = pd.DataFrame(ablation_summary_rows)
        if "none" in set(ablation_summary_df["ablation"]):
            base_val = float(ablation_summary_df.loc[ablation_summary_df["ablation"] == "none", "overall_mape"].iloc[0])
            ablation_summary_df["delta_vs_none"] = ablation_summary_df["overall_mape"] - base_val
        ablation_summary_df = ablation_summary_df.sort_values("overall_mape").reset_index(drop=True)
        ablation_summary_df.to_csv(inference_csv / "ablation_overall_summary.csv", index=False)

        if len(all_ablation_horizon_rows) > 0:
            ablation_horizon_df = pd.concat(all_ablation_horizon_rows, ignore_index=True)
            ablation_horizon_df.to_csv(inference_csv / "ablation_by_horizon.csv", index=False)
            plot_ablation_horizon(ablation_horizon_df, inference_plots / "ablation_by_horizon.png")
        else:
            ablation_horizon_df = pd.DataFrame()

        plot_ablation_overall(ablation_summary_df, inference_plots / "ablation_overall_comparison.png")

        # ----------------------------------------------------
        # No-training blending analysis on the standard payload
        # ----------------------------------------------------
        if main_none_long_df is not None and not main_none_long_df.empty:
            n_windows = int(main_none_long_df["window_idx"].nunique())
            calib_windows = max(1, int(round(args.calibration_fraction * n_windows)))
            calib_window_ids = set(sorted(main_none_long_df["window_idx"].unique())[:calib_windows])
            calib_df = main_none_long_df[main_none_long_df["window_idx"].isin(calib_window_ids)].copy()
            report_df = main_none_long_df[~main_none_long_df["window_idx"].isin(calib_window_ids)].copy()
            if report_df.empty:
                report_df = calib_df.copy()

            blend_grid = np.arange(0.0, 1.0 + 1e-9, args.blend_grid_step, dtype=np.float64)

            model_only_report = report_df.copy()
            model_only_report["blend_pred"] = model_only_report["pred"]
            prev_day_only_report = report_df.copy()
            prev_day_only_report["blend_pred"] = prev_day_only_report["prev_day_baseline"]

            blend_global_w = fit_blend_weights(calib_df, blend_grid, mode="global")
            blend_zone_w = fit_blend_weights(calib_df, blend_grid, mode="by_zone")
            blend_zone_h_w = fit_blend_weights(calib_df, blend_grid, mode="by_zone_horizon")

            blend_global_w.to_csv(inference_csv / "blend_weights_global.csv", index=False)
            blend_zone_w.to_csv(inference_csv / "blend_weights_by_zone.csv", index=False)
            blend_zone_h_w.to_csv(inference_csv / "blend_weights_by_zone_horizon.csv", index=False)

            blended_global_df = apply_blend_weights(report_df, blend_global_w, mode="global")
            blended_zone_df = apply_blend_weights(report_df, blend_zone_w, mode="by_zone")
            blended_zone_h_df = apply_blend_weights(report_df, blend_zone_h_w, mode="by_zone_horizon")

            blend_overall_parts = []
            blend_horizon_parts = []

            for label, bdf in [
                ("model_only", model_only_report),
                ("prev_day_only", prev_day_only_report),
                ("blend_global", blended_global_df),
                ("blend_by_zone", blended_zone_df),
                ("blend_by_zone_horizon", blended_zone_h_df),
            ]:
                ov, byh = summarize_blend_df(bdf, label)
                blend_overall_parts.append(ov)
                blend_horizon_parts.append(byh)

            blend_overall_df = pd.concat(blend_overall_parts, ignore_index=True)
            blend_horizon_df = pd.concat(blend_horizon_parts, ignore_index=True)

            blend_overall_df.to_csv(inference_csv / "blend_overall_report_metrics.csv", index=False)
            blend_horizon_df.to_csv(inference_csv / "blend_by_horizon_report_metrics.csv", index=False)

            fig, ax = plt.subplots(figsize=(10, 5))
            for label in blend_horizon_df["blend_label"].unique():
                sdf = blend_horizon_df[blend_horizon_df["blend_label"] == label].sort_values("forecast_horizon")
                ax.plot(sdf["forecast_horizon"], sdf["mape"], marker="o", label=label)
            ax.set_xlabel("Forecast horizon hour")
            ax.set_ylabel("MAPE (%)")
            ax.set_title("No-training blend comparison by horizon")
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(inference_plots / "blend_comparison_by_horizon.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 5))
            order = blend_overall_df.sort_values("mape")["blend_label"].tolist()
            tmp = blend_overall_df.set_index("blend_label").loc[order].reset_index()
            ax.bar(tmp["blend_label"], tmp["mape"])
            ax.set_ylabel("MAPE (%)")
            ax.set_title("No-training blend comparison (report windows)")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(inference_plots / "blend_comparison_overall.png", dpi=160)
            plt.close(fig)
        else:
            blend_overall_df = pd.DataFrame()

        write_inference_summary_text(
            ablation_summary_df=ablation_summary_df,
            blend_overall_df=blend_overall_df,
            out_path=inference_text / "inference_analysis_summary.txt",
        )

        save_json({
            "eval_years": eval_years,
            "device": str(device),
            "n_weather_vars": int(n_weather_vars),
            "used_model_history_hours": int(get_model_used_history(model, args.history_len)),
            "max_eval_windows": args.max_eval_windows,
            "midnight_only": bool(not args.all_hours),
        }, inference_root / "inference_run_summary.json")


    # --------------------------------------------------------
    # Advanced diagnostics for model-structure optimization
    # --------------------------------------------------------
    advanced_weather_plots = advanced_plots / "weather"
    advanced_weather_relevance_plots = advanced_plots / "weather_relevance"
    advanced_shared_factor_plots = advanced_plots / "shared_factors"
    advanced_shift_plots = advanced_plots / "distribution_shift"
    advanced_skill_plots = advanced_plots / "skill_atlas"
    advanced_residual_plots = advanced_plots / "residual_memory"
    advanced_volatility_plots = advanced_plots / "volatility"
    advanced_cohort_plots = advanced_plots / "extreme_cohorts"
    advanced_calibration_plots = advanced_plots / "calibration"
    for pth in [
        advanced_weather_plots,
        advanced_weather_relevance_plots,
        advanced_shared_factor_plots,
        advanced_shift_plots,
        advanced_skill_plots,
        advanced_residual_plots,
        advanced_volatility_plots,
        advanced_cohort_plots,
        advanced_calibration_plots,
    ]:
        ensure_dir(pth)

    print("Running advanced diagnostics...")

    # Dataset-only: year shift / baseline drift
    year_stats_df, year_pair_df = compute_year_distribution_shift(energy_only_df, zone_cols)
    year_stats_df.to_csv(advanced_csv / "yearly_zone_distribution_stats.csv", index=False)
    year_pair_df.to_csv(advanced_csv / "pairwise_year_distribution_shift.csv", index=False)

    baseline_year_df = compute_baseline_by_year(energy_only_df, zone_cols, args.history_len, args.future_len)
    baseline_year_df.to_csv(advanced_csv / "baseline_mape_by_year.csv", index=False)

    if not year_stats_df.empty:
        mean_pivot = year_stats_df.pivot(index="zone", columns="year", values="mean").sort_index().sort_index(axis=1)
        plot_heatmap(
            mean_pivot.to_numpy(dtype=np.float64),
            list(mean_pivot.columns),
            list(mean_pivot.index),
            title="Mean load by year and zone",
            out_path=advanced_shift_plots / "mean_load_by_year_zone.png",
            xlabel="Year",
            ylabel="Zone",
        )
    if not year_pair_df.empty:
        pair_label_df = year_pair_df.copy()
        pair_label_df["pair"] = pair_label_df["year_a"].astype(str) + "->" + pair_label_df["year_b"].astype(str)
        drift_pivot = pair_label_df.pivot(index="zone", columns="pair", values="approx_wasserstein").sort_index()
        plot_heatmap(
            drift_pivot.to_numpy(dtype=np.float64),
            list(drift_pivot.columns),
            list(drift_pivot.index),
            title="Approximate Wasserstein drift by year pair and zone",
            out_path=advanced_shift_plots / "wasserstein_shift_by_year_pair.png",
            xlabel="Year pair",
            ylabel="Zone",
        )
    if not baseline_year_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for baseline in baseline_year_df["baseline"].unique():
            sdf = baseline_year_df[baseline_year_df["baseline"] == baseline].sort_values("year")
            ax.plot(sdf["year"], sdf["overall_mape"], marker="o", label=baseline)
        ax.set_xlabel("Year")
        ax.set_ylabel("MAPE (%)")
        ax.set_title("Naive baseline drift by year")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(advanced_shift_plots / "baseline_mape_by_year.png", dpi=160)
        plt.close(fig)

    # Dataset-only: shared-factor structure
    raw_pca_df, raw_loading_df = compute_pca_summary(
        energy_only_df[zone_cols].to_numpy(dtype=np.float64),
        zone_cols,
        analysis_name="raw_demand",
        standardize=True,
    )
    deseasonalized_df = deseasonalize_energy_by_calendar(energy_df_with_cal, zone_cols)
    deseasonalized_pca_df, deseasonalized_loading_df = compute_pca_summary(
        deseasonalized_df[zone_cols].to_numpy(dtype=np.float64),
        zone_cols,
        analysis_name="deseasonalized_demand",
        standardize=True,
    )
    pd.concat([raw_pca_df, deseasonalized_pca_df], ignore_index=True).to_csv(advanced_csv / "shared_factor_explained_variance.csv", index=False)
    pd.concat([raw_loading_df, deseasonalized_loading_df], ignore_index=True).to_csv(advanced_csv / "shared_factor_loadings.csv", index=False)
    plot_scree(raw_pca_df, "Shared-factor scree | raw demand", advanced_shared_factor_plots / "scree_raw_demand.png")
    plot_scree(deseasonalized_pca_df, "Shared-factor scree | deseasonalized demand", advanced_shared_factor_plots / "scree_deseasonalized_demand.png")
    if not raw_loading_df.empty:
        cols = [c for c in raw_loading_df.columns if c.startswith("pc")][:3]
        plot_heatmap(
            raw_loading_df[cols].to_numpy(dtype=np.float64),
            cols,
            raw_loading_df["feature"].tolist(),
            title="Raw-demand factor loadings (first 3 PCs)",
            out_path=advanced_shared_factor_plots / "loadings_raw_demand.png",
            xlabel="Component",
            ylabel="Zone",
        )
    if not deseasonalized_loading_df.empty:
        cols = [c for c in deseasonalized_loading_df.columns if c.startswith("pc")][:3]
        plot_heatmap(
            deseasonalized_loading_df[cols].to_numpy(dtype=np.float64),
            cols,
            deseasonalized_loading_df["feature"].tolist(),
            title="Deseasonalized factor loadings (first 3 PCs)",
            out_path=advanced_shared_factor_plots / "loadings_deseasonalized_demand.png",
            xlabel="Component",
            ylabel="Zone",
        )

    # Dataset-only: advanced weather diagnostics
    sampled_weather_feature_df = build_sampled_weather_feature_frame(
        energy_only_df,
        zone_cols,
        weather_lookup,
        weather_var_names,
        weather_stride=max(1, args.weather_stride),
        max_weather_files=args.max_weather_files,
        tile_grid=(6, 6),
    )
    sampled_weather_feature_df.to_csv(advanced_csv / "sampled_weather_feature_frame.csv", index=False)

    weather_redundancy_summary_df, weather_redundancy_pairs_df, weather_corr_matrix = compute_weather_channel_redundancy(sampled_weather_feature_df)
    weather_redundancy_summary_df.to_csv(advanced_csv / "weather_redundancy_summary.csv", index=False)
    weather_redundancy_pairs_df.to_csv(advanced_csv / "weather_redundancy_pairs.csv", index=False)
    if isinstance(weather_corr_matrix, pd.DataFrame) and not weather_corr_matrix.empty:
        weather_corr_matrix.to_csv(advanced_csv / "weather_channel_correlation_matrix.csv")
        plot_heatmap(
            weather_corr_matrix.to_numpy(dtype=np.float64),
            weather_corr_matrix.columns.tolist(),
            weather_corr_matrix.index.tolist(),
            title="Weather spatial-mean channel correlation",
            out_path=advanced_weather_plots / "weather_channel_correlation_heatmap.png",
            xlabel="Channel",
            ylabel="Channel",
        )
    if not weather_redundancy_pairs_df.empty:
        top_pairs = weather_redundancy_pairs_df.head(15)
        fig, ax = plt.subplots(figsize=(11, 5))
        labels = [f"{a} | {b}" for a, b in zip(top_pairs["feature_a"], top_pairs["feature_b"])]
        ax.bar(labels, top_pairs["abs_corr"])
        ax.set_ylabel("Absolute correlation")
        ax.set_title("Top redundant weather channel pairs")
        ax.tick_params(axis="x", rotation=75)
        fig.tight_layout()
        fig.savefig(advanced_weather_plots / "top_weather_redundancy_pairs.png", dpi=160)
        plt.close(fig)
    if isinstance(weather_corr_matrix, pd.DataFrame) and not weather_corr_matrix.empty:
        weather_pca_df, _ = compute_pca_summary(
            sampled_weather_feature_df[[c for c in sampled_weather_feature_df.columns if c.startswith("mean__")]].to_numpy(dtype=np.float64),
            [c for c in sampled_weather_feature_df.columns if c.startswith("mean__")],
            analysis_name="weather_spatial_mean_features",
            standardize=True,
        )
        weather_pca_df.to_csv(advanced_csv / "weather_feature_pca_explained_variance.csv", index=False)
        plot_scree(weather_pca_df, "Weather feature scree | spatial means", advanced_weather_plots / "weather_feature_scree.png")
    else:
        weather_pca_df = pd.DataFrame()

    tile_corr_df, weather_relevance_summary_df = compute_zone_specific_weather_relevance(sampled_weather_feature_df, zone_cols)
    tile_corr_df.to_csv(advanced_csv / "weather_relevance_tile_correlations.csv", index=False)
    weather_relevance_summary_df.to_csv(advanced_csv / "weather_relevance_summary.csv", index=False)
    plot_weather_relevance_heatmaps(
        tile_corr_df=tile_corr_df,
        summary_df=weather_relevance_summary_df,
        out_dir=advanced_weather_relevance_plots,
        top_k_channels=3,
    )

    calibration_year_df = pd.DataFrame()
    skill_outputs: Dict[str, pd.DataFrame] = {}
    residual_memory_summary_df = pd.DataFrame()
    cohort_df = pd.DataFrame()

    # Inference-dependent advanced diagnostics
    if main_none_long_df is not None and not main_none_long_df.empty:
        advanced_long_df = ensure_long_df_time_columns(main_none_long_df)
        advanced_long_df.to_csv(advanced_csv / "inference_main_none_long_df.csv", index=False)

        skill_outputs = compute_skill_atlas(advanced_long_df)
        for name, df in skill_outputs.items():
            df.to_csv(advanced_csv / f"skill_atlas_{name}.csv", index=False)
        plot_skill_atlas(skill_outputs, advanced_skill_plots)

        hourly_residual_df = build_hourly_residual_frame(advanced_long_df)
        hourly_residual_df.to_csv(advanced_csv / "hourly_residual_frame.csv", index=False)

        residual_acf_df, residual_memory_summary_df = compute_residual_memory_analysis(hourly_residual_df, zone_cols)
        residual_acf_df.to_csv(advanced_csv / "residual_memory_acf.csv", index=False)
        residual_memory_summary_df.to_csv(advanced_csv / "residual_memory_summary.csv", index=False)
        plot_residual_memory(residual_acf_df, advanced_residual_plots)

        model_residual_pivot = hourly_residual_df.pivot(index="timestamp_utc", columns="zone", values="error").sort_index().reindex(columns=zone_cols)
        base_residual_pivot = hourly_residual_df.pivot(index="timestamp_utc", columns="zone", values="baseline_error").sort_index().reindex(columns=zone_cols)
        model_residual_pca_df, model_residual_loading_df = compute_pca_summary(
            model_residual_pivot.to_numpy(dtype=np.float64),
            zone_cols,
            analysis_name="hourly_model_residual",
            standardize=True,
        )
        base_residual_pca_df, base_residual_loading_df = compute_pca_summary(
            base_residual_pivot.to_numpy(dtype=np.float64),
            zone_cols,
            analysis_name="hourly_prev_day_residual",
            standardize=True,
        )
        pd.concat([model_residual_pca_df, base_residual_pca_df], ignore_index=True).to_csv(advanced_csv / "residual_shared_factor_explained_variance.csv", index=False)
        pd.concat([model_residual_loading_df, base_residual_loading_df], ignore_index=True).to_csv(advanced_csv / "residual_shared_factor_loadings.csv", index=False)
        plot_scree(model_residual_pca_df, "Shared-factor scree | model residual", advanced_shared_factor_plots / "scree_model_residual.png")
        plot_scree(base_residual_pca_df, "Shared-factor scree | prev-day residual", advanced_shared_factor_plots / "scree_prev_day_residual.png")

        zone_h_vol_df, month_hour_vol_df, weekend_vol_df = compute_conditional_error_volatility(advanced_long_df)
        zone_h_vol_df.to_csv(advanced_csv / "conditional_error_volatility_zone_horizon.csv", index=False)
        month_hour_vol_df.to_csv(advanced_csv / "conditional_error_volatility_month_hour.csv", index=False)
        weekend_vol_df.to_csv(advanced_csv / "conditional_error_volatility_weekend.csv", index=False)
        plot_conditional_error_volatility(zone_h_vol_df, month_hour_vol_df, advanced_volatility_plots)

        cohort_df = compute_extreme_event_cohorts(advanced_long_df)
        cohort_df.to_csv(advanced_csv / "extreme_event_cohort_metrics.csv", index=False)
        plot_extreme_event_cohorts(cohort_df, advanced_cohort_plots)

        calibration_year_df, calibration_month_df, calibration_decile_df = compute_prediction_calibration(advanced_long_df)
        calibration_year_df.to_csv(advanced_csv / "prediction_calibration_by_year.csv", index=False)
        calibration_month_df.to_csv(advanced_csv / "prediction_calibration_by_month.csv", index=False)
        calibration_decile_df.to_csv(advanced_csv / "prediction_calibration_by_target_decile.csv", index=False)
        plot_prediction_calibration(calibration_year_df, calibration_month_df, calibration_decile_df, advanced_calibration_plots)

        save_json({
            "eval_years": eval_years_for_advanced,
            "source": "main none ablation payload",
        }, advanced_root / "inference_backed_advanced_diagnostics.json")
    else:
        with open(advanced_text / "inference_dependent_diagnostics_note.txt", "w", encoding="utf-8") as f:
            f.write(
                "Inference-dependent advanced diagnostics were skipped because --run-inference-analysis was not used or no standard payload was available.\n"
            )

    write_advanced_summary(
        out_path=advanced_text / "advanced_diagnostics_summary.txt",
        year_pair_df=year_pair_df,
        shared_factor_raw_df=raw_pca_df,
        weather_relevance_summary_df=weather_relevance_summary_df,
        skill_zone_h_df=skill_outputs.get("by_zone_horizon", pd.DataFrame()),
        residual_memory_summary_df=residual_memory_summary_df,
        cohort_df=cohort_df,
        weather_redundancy_summary_df=weather_redundancy_summary_df,
        calibration_year_df=calibration_year_df,
    )


    print("Done.")
    print(f"Outputs written to: {out_dir}")
    print(f"CSV files: {csv_dir}")
    print(f"Plots:     {plot_dir}")
    print(f"Notes:     {text_dir}")
    print(f"Advanced diagnostics: {advanced_root}")
    if args.run_inference_analysis:
        print(f"Inference-only analysis: {out_dir / 'inference_only_analysis'}")


if __name__ == "__main__":
    main()
