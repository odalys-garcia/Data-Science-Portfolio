from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm.auto import tqdm

try:
    from generate_dataset_skipbad_debugged import EnergyWeatherDataset, compute_normalization_stats, prebuild_manifests
except ImportError:
    try:
        from generate_dataset_skipbad import EnergyWeatherDataset, compute_normalization_stats, prebuild_manifests
    except ImportError:
        from generate_dataset import EnergyWeatherDataset, compute_normalization_stats  # type: ignore
        prebuild_manifests = None

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


WEATHER_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data"
ENERGY_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data"
DEFAULT_TRAIN_YEARS = [2019, 2020, 2021, 2022]
DEFAULT_VAL_YEARS = [2023]
DEFAULT_EVAL_YEARS = [2024]
DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
DEFAULT_REPORT_ROOT = Path(__file__).resolve().parent / "eval_reports"
SELECTED_VARS: List[str] = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()


def parse_years(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def sanitize_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


def infer_checkpoint_path(model_name: Optional[str], explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        return Path(explicit_path)
    if not model_name:
        return None
    safe = sanitize_name(model_name)
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / f"{safe}.pth",
        script_dir / "checkpoints" / f"{safe}_best.pth",
        script_dir / "checkpoints" / f"{safe}_latest.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def unnormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.view(1, 1, -1).to(tensor.device)
    std = std.view(1, 1, -1).to(tensor.device)
    return tensor * std + mean


def calculate_mape(preds: np.ndarray, targets: np.ndarray, epsilon: float = 1e-6) -> float:
    denom = np.clip(np.abs(targets), epsilon, None)
    return float(np.mean(np.abs(preds - targets) / denom) * 100.0)


def calculate_smape(preds: np.ndarray, targets: np.ndarray, epsilon: float = 1e-6) -> float:
    denom = np.clip(np.abs(preds) + np.abs(targets), epsilon, None)
    return float(np.mean(2.0 * np.abs(preds - targets) / denom) * 100.0)


def safe_r2(targets: np.ndarray, preds: np.ndarray) -> float:
    ss_res = float(np.sum((targets - preds) ** 2))
    centered = targets - np.mean(targets)
    ss_tot = float(np.sum(centered ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def fit_calibration(targets: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(targets, dtype=np.float64).reshape(-1)
    y = np.asarray(preds, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    x_mean = x.mean()
    y_mean = y.mean()
    var_x = ((x - x_mean) ** 2).mean()
    if var_x <= 1e-12:
        return float("nan"), float(y_mean), float("nan")
    cov_xy = ((x - x_mean) * (y - y_mean)).mean()
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    corr_num = cov_xy
    corr_den = math.sqrt(max(var_x, 0.0) * max(((y - y_mean) ** 2).mean(), 0.0))
    corr = corr_num / corr_den if corr_den > 1e-12 else float("nan")
    return float(slope), float(intercept), float(corr)


def summarize_arrays(targets: np.ndarray, preds: np.ndarray, baseline: Optional[np.ndarray] = None) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    abs_err = np.abs(preds - targets)
    sq_err = (preds - targets) ** 2
    denom = np.clip(np.abs(targets), 1e-6, None)

    out: Dict[str, float] = {
        "count": float(targets.size),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(sq_err.mean())),
        "mape": float((abs_err / denom).mean() * 100.0),
        "smape": calculate_smape(preds, targets),
        "wape": float(abs_err.sum() / np.clip(np.abs(targets).sum(), 1e-6, None) * 100.0),
        "bias": float((preds - targets).mean()),
        "mean_target": float(targets.mean()),
        "mean_pred": float(preds.mean()),
        "median_ae": float(np.median(abs_err)),
        "p90_ae": float(np.percentile(abs_err, 90)),
        "p95_ae": float(np.percentile(abs_err, 95)),
        "r2": safe_r2(targets, preds),
    }
    slope, intercept, corr = fit_calibration(targets, preds)
    out["calibration_slope"] = slope
    out["calibration_intercept"] = intercept
    out["corr"] = corr

    if baseline is not None:
        baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
        b_abs_err = np.abs(baseline - targets)
        b_sq_err = (baseline - targets) ** 2
        out["baseline_mae"] = float(b_abs_err.mean())
        out["baseline_rmse"] = float(np.sqrt(b_sq_err.mean()))
        out["baseline_mape"] = float((b_abs_err / denom).mean() * 100.0)
        out["baseline_smape"] = calculate_smape(baseline, targets)
        out["baseline_wape"] = float(b_abs_err.sum() / np.clip(np.abs(targets).sum(), 1e-6, None) * 100.0)
        out["baseline_bias"] = float((baseline - targets).mean())
        out["mape_gain_vs_baseline_pp"] = out["baseline_mape"] - out["mape"]
        out["mae_gain_vs_baseline"] = out["baseline_mae"] - out["mae"]
        out["rmse_gain_vs_baseline"] = out["baseline_rmse"] - out["rmse"]
    return out


def group_metrics(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = df.groupby(by, dropna=False, sort=True)
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = {col: val for col, val in zip(by, keys)}
        row.update(summarize_arrays(g["actual"].to_numpy(), g["pred"].to_numpy(), g["baseline"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def make_plot(x, y_model, y_baseline, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y_model, marker="o", label="model")
    if y_baseline is not None:
        ax.plot(x, y_baseline, marker="o", label="prev_day_baseline")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_bar_plot(labels, values, baseline_values, ylabel: str, title: str, out_path: Path, rotate: bool = False) -> None:
    x = np.arange(len(labels))
    width = 0.4
    fig = plt.figure(figsize=(max(10, 0.45 * len(labels)), 5.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x - width / 2, values, width=width, label="model")
    if baseline_values is not None:
        ax.bar(x + width / 2, baseline_values, width=width, label="prev_day_baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if rotate else 0, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_scatter_plot(actual: np.ndarray, pred: np.ndarray, out_path: Path, max_points: int = 50000) -> None:
    actual = actual.reshape(-1)
    pred = pred.reshape(-1)
    n = actual.size
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        actual = actual[idx]
        pred = pred[idx]
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(actual, pred, s=4, alpha=0.25)
    lo = min(float(actual.min()), float(pred.min()))
    hi = max(float(actual.max()), float(pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title("Predicted vs actual")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_histogram(values: np.ndarray, xlabel: str, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(values.reshape(-1), bins=80)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_direction_hints(
    overall: Dict[str, float],
    per_horizon: pd.DataFrame,
    per_zone: pd.DataFrame,
    by_hour: pd.DataFrame,
    difficulty: pd.DataFrame,
    peaks: pd.DataFrame,
) -> List[str]:
    hints: List[str] = []

    baseline_mape = overall.get("baseline_mape", float("nan"))
    model_mape = overall.get("mape", float("nan"))
    if np.isfinite(baseline_mape) and np.isfinite(model_mape):
        gain_pp = baseline_mape - model_mape
        rel_gain = gain_pp / max(baseline_mape, 1e-6)
        if rel_gain < 0.10:
            hints.append(
                "The model is only modestly better than the prev-day baseline overall. That usually means the next gains are more likely to come from improving residual modeling on hard hours rather than just scaling the current backbone."
            )
        elif gain_pp < 0:
            hints.append(
                "The prev-day baseline is outperforming the model overall. Before increasing capacity, check regularization, learning-rate schedule, and whether the residual head is overfitting easy windows while hurting simple persistence behavior."
            )

    if not per_horizon.empty:
        first = float(per_horizon.iloc[0]["mape"])
        last = float(per_horizon.iloc[-1]["mape"])
        if last > first * 1.20:
            hints.append(
                "Error grows materially with forecast horizon. That points toward weak long-range future context usage, so likely directions are stronger horizon-specific decoding, autoregressive refinement, or explicit multi-scale future summaries."
            )
        if "mape_gain_vs_baseline_pp" in per_horizon.columns and (per_horizon["mape_gain_vs_baseline_pp"] < 0).any():
            bad_h = per_horizon.loc[per_horizon["mape_gain_vs_baseline_pp"] < 0, "horizon"].tolist()
            hints.append(
                f"The model loses to the prev-day baseline at some horizons ({bad_h[:8]}{'...' if len(bad_h) > 8 else ''}). Focus on horizon-conditioned losses or decoders instead of globally scaling the architecture."
            )

    if not peaks.empty:
        overall_peak = peaks.loc[peaks["slice"] == "top_5pct_actual"]
        if not overall_peak.empty:
            peak_mape = float(overall_peak.iloc[0]["mape"])
            if np.isfinite(model_mape) and peak_mape > model_mape * 1.5:
                hints.append(
                    "Peak-demand error is much worse than overall error. The clearest next direction is a peak-aware objective: weighted loss on high-load hours, residual scaling, or a separate extreme-event head/calibration pass."
                )

    if not difficulty.empty:
        worst = difficulty.sort_values("difficulty_bin").iloc[-1]
        easiest = difficulty.sort_values("difficulty_bin").iloc[0]
        if float(worst["mape"]) > float(easiest["mape"]) * 1.8:
            hints.append(
                "Hard residual windows drive a disproportionately large share of the error. Since the architecture is explicitly residual-over-prev-day, target the training distribution or loss toward large |target - baseline| cases."
            )

    if not per_zone.empty:
        zone_spread = float(per_zone["mape"].max() - per_zone["mape"].min())
        if zone_spread > 1.0:
            worst_zone = str(per_zone.sort_values("mape", ascending=False).iloc[0]["zone_name"])
            hints.append(
                f"Performance varies meaningfully by zone, with {worst_zone} among the weakest. That suggests value in stronger zone-specific capacity, per-zone calibration, or zone-clustered heads instead of only shared global layers."
            )

    if not by_hour.empty:
        hour_spread = float(by_hour["mape"].max() - by_hour["mape"].min())
        if hour_spread > 1.0:
            worst_hour = int(by_hour.sort_values("mape", ascending=False).iloc[0]["hour_of_day"])
            hints.append(
                f"There is a strong hour-of-day pattern in the errors, especially around hour {worst_hour}. That usually means calendar conditioning is still under-expressive relative to the load cycle, or that certain ramp windows need targeted weighting."
            )

    bias = abs(float(overall.get("bias", 0.0)))
    mean_target = max(abs(float(overall.get("mean_target", 0.0))), 1e-6)
    if bias / mean_target > 0.03:
        hints.append(
            "The model shows noticeable mean bias. A lightweight post-hoc calibration layer or bias correction on top of the current checkpoint may give cheap gains before changing the core architecture."
        )

    if not hints:
        hints.append(
            "The diagnostics do not show a single dominant failure mode. The most promising next step is to rank candidate changes by where they improve baseline-relative error: horizon-specific, zone-specific, and peak-specific slices."
        )
    return hints


def build_peak_slices(df: pd.DataFrame) -> pd.DataFrame:
    actual = df["actual"].to_numpy()
    q90 = float(np.quantile(actual, 0.90))
    q95 = float(np.quantile(actual, 0.95))
    q99 = float(np.quantile(actual, 0.99))
    slices = [
        ("all_points", np.ones(len(df), dtype=bool)),
        ("top_10pct_actual", actual >= q90),
        ("top_5pct_actual", actual >= q95),
        ("top_1pct_actual", actual >= q99),
    ]
    rows: List[Dict[str, object]] = []
    for name, mask in slices:
        g = df.loc[mask]
        row: Dict[str, object] = {"slice": name}
        row.update(summarize_arrays(g["actual"].to_numpy(), g["pred"].to_numpy(), g["baseline"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_difficulty_bins(df: pd.DataFrame, num_bins: int = 5) -> pd.DataFrame:
    difficulty = pd.Series(np.abs(df["actual"].to_numpy() - df["baseline"].to_numpy()))
    try:
        bins = pd.qcut(difficulty, q=num_bins, labels=[f"Q{i+1}" for i in range(num_bins)], duplicates="drop")
    except ValueError:
        bins = pd.Series(["Q1"] * len(df), index=df.index)
    temp = df.copy()
    temp["difficulty_bin"] = bins.astype(str)
    out = group_metrics(temp, ["difficulty_bin"])
    order = {f"Q{i+1}": i for i in range(num_bins)}
    return out.sort_values("difficulty_bin", key=lambda s: s.map(order).fillna(999)).reset_index(drop=True)


def choose_eval_indices(
    dataset: EnergyWeatherDataset,
    start_date: str = "",
    end_date: str = "",
    max_windows: int = 0,
    sample_frac: float = 1.0,
    sample_seed: int = 0,
    sample_mode: str = "stratified_month",
) -> List[int]:
    total = len(dataset)
    if total == 0:
        return []

    all_indices = np.arange(total, dtype=np.int64)
    future_start_timestamps = []
    for ds_idx in all_indices:
        start_idx = int(dataset.valid_start_indices[int(ds_idx)])
        fut_start = dataset.energy_df.index[start_idx + dataset.S]
        future_start_timestamps.append(fut_start)
    future_start_timestamps = pd.to_datetime(pd.Index(future_start_timestamps), utc=True)

    mask = np.ones(total, dtype=bool)
    if start_date:
        start_ts = pd.Timestamp(start_date)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        mask &= (future_start_timestamps >= start_ts).to_numpy()
    if end_date:
        end_ts = pd.Timestamp(end_date)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        mask &= (future_start_timestamps <= end_ts).to_numpy()

    candidates = all_indices[mask]
    if candidates.size == 0:
        return []

    desired = candidates.size
    if sample_frac > 0.0 and sample_frac < 1.0:
        desired = min(desired, max(1, int(round(candidates.size * float(sample_frac)))))
    if max_windows > 0:
        desired = min(desired, int(max_windows))

    if desired >= candidates.size:
        return candidates.tolist()

    rng = np.random.default_rng(int(sample_seed))
    sample_mode = str(sample_mode).strip().lower()

    if sample_mode == "sequential":
        selected = candidates[:desired]
    elif sample_mode == "random":
        selected = np.sort(rng.choice(candidates, size=desired, replace=False))
    else:
        ts = future_start_timestamps[candidates]
        frame = pd.DataFrame({"dataset_idx": candidates, "month": ts.month.to_numpy()})
        if sample_mode == "stratified_month_hour":
            frame["hour"] = ts.hour.to_numpy()
            group_cols = ["month", "hour"]
        else:
            group_cols = ["month"]

        groups = list(frame.groupby(group_cols, sort=True, dropna=False))
        if len(groups) == 0:
            selected = candidates[:desired]
        else:
            total_candidates = len(frame)
            remaining = desired
            picked_chunks = []
            leftovers = []

            for _, group in groups:
                g_idx = group["dataset_idx"].to_numpy(dtype=np.int64)
                proportional = int(np.floor(desired * (len(group) / float(total_candidates))))
                take = min(len(g_idx), proportional)
                if take > 0:
                    chosen = rng.choice(g_idx, size=take, replace=False)
                    picked_chunks.append(chosen)
                    used = set(int(x) for x in chosen.tolist())
                    rem = np.array([x for x in g_idx if int(x) not in used], dtype=np.int64)
                else:
                    rem = g_idx
                remaining -= take
                if rem.size > 0:
                    leftovers.append(rem)

            if remaining > 0 and leftovers:
                pool = np.concatenate(leftovers)
                if remaining >= pool.size:
                    picked_chunks.append(pool)
                else:
                    picked_chunks.append(rng.choice(pool, size=remaining, replace=False))

            selected = np.sort(np.unique(np.concatenate(picked_chunks))) if picked_chunks else candidates[:desired]
            if selected.size > desired:
                selected = np.sort(rng.choice(selected, size=desired, replace=False))
            elif selected.size < desired:
                missing = desired - selected.size
                available = np.setdiff1d(candidates, selected, assume_unique=False)
                if missing > 0 and available.size > 0:
                    fill = available if missing >= available.size else rng.choice(available, size=missing, replace=False)
                    selected = np.sort(np.concatenate([selected, fill]))

    return [int(x) for x in np.sort(selected)]


def build_eval_subset(dataset: EnergyWeatherDataset, selected_indices: List[int]) -> Tuple[torch.utils.data.Dataset, List[int]]:
    if not selected_indices:
        raise RuntimeError("No evaluation windows were selected. Widen the date range or increase max_windows/sample_frac.")
    if len(selected_indices) == len(dataset):
        return dataset, selected_indices
    return Subset(dataset, selected_indices), selected_indices


def collect_point_dataframe(
    model: torch.nn.Module,
    dataset: EnergyWeatherDataset,
    loader: DataLoader,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    amp_enabled: bool,
    source_dataset_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    model.eval()
    source_dataset_indices = list(source_dataset_indices) if source_dataset_indices is not None else list(range(len(dataset)))
    sample_cursor = 0

    sample_id_list: List[np.ndarray] = []
    timestamp_list: List[np.ndarray] = []
    horizon_list: List[np.ndarray] = []
    zone_idx_list: List[np.ndarray] = []
    hour_list: List[np.ndarray] = []
    dow_list: List[np.ndarray] = []
    month_list: List[np.ndarray] = []
    actual_list: List[np.ndarray] = []
    pred_list: List[np.ndarray] = []
    baseline_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            weather_seq, hist_demand, hist_cal, fut_cal, target_demand = [b.to(DEVICE, non_blocking=True) for b in batch]
            with autocast(device_type=DEVICE.type, enabled=amp_enabled):
                pred_norm = model(weather_seq, hist_demand, hist_cal, fut_cal)

            pred_raw = unnormalize(pred_norm, norm_mean.to(DEVICE), norm_std.to(DEVICE)).detach().cpu().numpy().astype(np.float32)
            target_raw = unnormalize(target_demand, norm_mean.to(DEVICE), norm_std.to(DEVICE)).detach().cpu().numpy().astype(np.float32)
            baseline_raw = unnormalize(hist_demand[:, -dataset.F :, :], norm_mean.to(DEVICE), norm_std.to(DEVICE)).detach().cpu().numpy().astype(np.float32)

            batch_size = pred_raw.shape[0]
            horizons = np.arange(dataset.F, dtype=np.int16)
            zones = np.arange(dataset.num_zones, dtype=np.int16)

            for j in range(batch_size):
                ds_idx = int(source_dataset_indices[sample_cursor + j])
                start_idx = int(dataset.valid_start_indices[ds_idx])
                fut_index = dataset.energy_df.index[start_idx + dataset.S : start_idx + dataset.S + dataset.F]
                fut_dt64 = fut_index.to_numpy(dtype="datetime64[ns]")
                fut_hours = fut_index.hour.to_numpy(dtype=np.int16)
                fut_dow = fut_index.dayofweek.to_numpy(dtype=np.int16)
                fut_month = fut_index.month.to_numpy(dtype=np.int16)

                sample_id_arr = np.full(dataset.F * dataset.num_zones, ds_idx, dtype=np.int32)
                timestamp_arr = np.repeat(fut_dt64, dataset.num_zones)
                horizon_arr = np.repeat(horizons, dataset.num_zones)
                zone_arr = np.tile(zones, dataset.F)
                hour_arr = np.repeat(fut_hours, dataset.num_zones)
                dow_arr = np.repeat(fut_dow, dataset.num_zones)
                month_arr = np.repeat(fut_month, dataset.num_zones)

                sample_id_list.append(sample_id_arr)
                timestamp_list.append(timestamp_arr)
                horizon_list.append(horizon_arr)
                zone_idx_list.append(zone_arr)
                hour_list.append(hour_arr)
                dow_list.append(dow_arr)
                month_list.append(month_arr)
                actual_list.append(target_raw[j].reshape(-1))
                pred_list.append(pred_raw[j].reshape(-1))
                baseline_list.append(baseline_raw[j].reshape(-1))

            sample_cursor += batch_size

    df = pd.DataFrame(
        {
            "sample_id": np.concatenate(sample_id_list),
            "timestamp": np.concatenate(timestamp_list),
            "horizon": np.concatenate(horizon_list),
            "zone_idx": np.concatenate(zone_idx_list),
            "hour_of_day": np.concatenate(hour_list),
            "day_of_week": np.concatenate(dow_list),
            "month": np.concatenate(month_list),
            "actual": np.concatenate(actual_list),
            "pred": np.concatenate(pred_list),
            "baseline": np.concatenate(baseline_list),
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["zone_name"] = [dataset.zone_names[int(i)] for i in df["zone_idx"].to_numpy()]
    df["residual"] = df["pred"] - df["actual"]
    df["abs_err"] = np.abs(df["residual"])
    df["ape"] = df["abs_err"] / np.clip(np.abs(df["actual"]), 1e-6, None) * 100.0
    df["baseline_abs_err"] = np.abs(df["baseline"] - df["actual"])
    df["baseline_ape"] = df["baseline_abs_err"] / np.clip(np.abs(df["actual"]), 1e-6, None) * 100.0
    df["baseline_gap"] = np.abs(df["actual"] - df["baseline"])
    return df


def save_reports(df: pd.DataFrame, out_dir: Path) -> None:
    metrics_dir = out_dir / "metrics"
    plots_dir = out_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    overall = summarize_arrays(df["actual"].to_numpy(), df["pred"].to_numpy(), df["baseline"].to_numpy())
    save_json(metrics_dir / "overall_metrics.json", overall)
    pd.DataFrame([overall]).to_csv(metrics_dir / "overall_metrics.csv", index=False)

    per_zone = group_metrics(df, ["zone_idx", "zone_name"]).sort_values("mape", ascending=False)
    per_horizon = group_metrics(df, ["horizon"]).sort_values("horizon")
    per_zone_horizon = group_metrics(df, ["zone_idx", "zone_name", "horizon"]).sort_values(["zone_idx", "horizon"])
    by_hour = group_metrics(df, ["hour_of_day"]).sort_values("hour_of_day")
    by_dow = group_metrics(df, ["day_of_week"]).sort_values("day_of_week")
    by_month = group_metrics(df, ["month"]).sort_values("month")
    difficulty = compute_difficulty_bins(df, num_bins=5)
    peaks = build_peak_slices(df)

    sample_metrics = group_metrics(df, ["sample_id"])
    worst_windows = sample_metrics.sort_values("mape", ascending=False).head(200)

    per_zone.to_csv(metrics_dir / "per_zone_metrics.csv", index=False)
    per_horizon.to_csv(metrics_dir / "per_horizon_metrics.csv", index=False)
    per_zone_horizon.to_csv(metrics_dir / "per_zone_horizon_metrics.csv", index=False)
    by_hour.to_csv(metrics_dir / "by_hour_metrics.csv", index=False)
    by_dow.to_csv(metrics_dir / "by_day_of_week_metrics.csv", index=False)
    by_month.to_csv(metrics_dir / "by_month_metrics.csv", index=False)
    difficulty.to_csv(metrics_dir / "difficulty_bin_metrics.csv", index=False)
    peaks.to_csv(metrics_dir / "peak_slice_metrics.csv", index=False)
    worst_windows.to_csv(metrics_dir / "worst_windows.csv", index=False)

    direction_hints = build_direction_hints(overall, per_horizon, per_zone, by_hour, difficulty, peaks)
    hints_payload = {
        "summary": overall,
        "recommended_directions": direction_hints,
    }
    save_json(metrics_dir / "direction_hints.json", hints_payload)
    with (metrics_dir / "direction_hints.txt").open("w", encoding="utf-8") as f:
        for i, hint in enumerate(direction_hints, start=1):
            f.write(f"{i}. {hint}\n")

    make_plot(
        per_horizon["horizon"].to_numpy(),
        per_horizon["mape"].to_numpy(),
        per_horizon["baseline_mape"].to_numpy(),
        xlabel="horizon step",
        ylabel="MAPE (%)",
        title="MAPE by forecast horizon",
        out_path=plots_dir / "horizon_mape.png",
    )
    make_plot(
        by_hour["hour_of_day"].to_numpy(),
        by_hour["mape"].to_numpy(),
        by_hour["baseline_mape"].to_numpy(),
        xlabel="hour of day",
        ylabel="MAPE (%)",
        title="MAPE by hour of day",
        out_path=plots_dir / "hour_of_day_mape.png",
    )
    make_plot(
        by_month["month"].to_numpy(),
        by_month["mape"].to_numpy(),
        by_month["baseline_mape"].to_numpy(),
        xlabel="month",
        ylabel="MAPE (%)",
        title="MAPE by month",
        out_path=plots_dir / "month_mape.png",
    )
    make_bar_plot(
        labels=per_zone["zone_name"].astype(str).tolist(),
        values=per_zone["mape"].to_numpy(),
        baseline_values=per_zone["baseline_mape"].to_numpy(),
        ylabel="MAPE (%)",
        title="MAPE by zone",
        out_path=plots_dir / "zone_mape.png",
        rotate=True,
    )
    make_bar_plot(
        labels=difficulty["difficulty_bin"].astype(str).tolist(),
        values=difficulty["mape"].to_numpy(),
        baseline_values=difficulty["baseline_mape"].to_numpy(),
        ylabel="MAPE (%)",
        title="MAPE by residual difficulty bin",
        out_path=plots_dir / "difficulty_mape.png",
    )
    make_bar_plot(
        labels=peaks["slice"].astype(str).tolist(),
        values=peaks["mape"].to_numpy(),
        baseline_values=peaks["baseline_mape"].to_numpy(),
        ylabel="MAPE (%)",
        title="Overall vs peak-load slices",
        out_path=plots_dir / "peak_slice_mape.png",
        rotate=True,
    )
    make_scatter_plot(df["actual"].to_numpy(), df["pred"].to_numpy(), plots_dir / "pred_vs_actual.png")
    make_histogram(df["residual"].to_numpy(), xlabel="prediction residual", title="Residual distribution", out_path=plots_dir / "residual_histogram.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detailed evaluation and diagnostic reporting for the weather-to-energy model.")
    parser.add_argument("--weather-dir", type=str, default=WEATHER_DIR)
    parser.add_argument("--energy-dir", type=str, default=ENERGY_DIR)
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--train-years", type=str, default=",".join(map(str, DEFAULT_TRAIN_YEARS)))
    parser.add_argument("--val-years", type=str, default=",".join(map(str, DEFAULT_VAL_YEARS)))
    parser.add_argument("--eval-years", type=str, default=",".join(map(str, DEFAULT_EVAL_YEARS)))
    parser.add_argument("--seq-len-hist", type=int, default=24)
    parser.add_argument("--seq-len-fut", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--manifest-dir", type=str, default=str(DEFAULT_MANIFEST_DIR))
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--skip-prebuild-manifests", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--save-point-records", action="store_true")
    parser.add_argument("--report-dir", type=str, default="")
    parser.add_argument("--start-date", type=str, default="")
    parser.add_argument("--end-date", type=str, default="")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="stratified_month",
        choices=["sequential", "random", "stratified_month", "stratified_month_hour"],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    weather_dir = Path(args.weather_dir)
    energy_dir = Path(args.energy_dir)
    train_years = parse_years(args.train_years)
    val_years = parse_years(args.val_years)
    eval_years = parse_years(args.eval_years)
    manifest_dir = Path(args.manifest_dir)
    use_manifest = not args.no_manifest
    rebuild_manifest = bool(args.rebuild_manifest)
    amp_enabled = (DEVICE.type == "cuda") and (not args.disable_amp)

    ckpt_path = infer_checkpoint_path(args.model_name or None, args.checkpoint_path or None)
    metadata: Dict[str, object] = {
        "load_checkpoint": True,
        "return_denormalized": False,
    }
    if ckpt_path is not None:
        metadata["checkpoint_path"] = str(ckpt_path)

    ckpt = None
    if ckpt_path is not None and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if use_manifest and prebuild_manifests is not None and not args.skip_prebuild_manifests:
        prebuild_manifests(
            weather_dir=weather_dir,
            energy_dir=energy_dir,
            years_groups=[train_years, val_years, eval_years],
            selected_vars=SELECTED_VARS,
            seq_configs=[(args.seq_len_hist, args.seq_len_fut)],
            manifest_dir=manifest_dir,
            rebuild_manifest=rebuild_manifest,
            verbose=True,
        )

    norm_mean = None
    norm_std = None
    if isinstance(ckpt, dict) and ("norm_mean" in ckpt) and ("norm_std" in ckpt):
        norm_mean = ckpt["norm_mean"].detach().float()
        norm_std = ckpt["norm_std"].detach().float()
        print("[*] Using normalization stats from checkpoint.")
    else:
        print("[WARNING] Checkpoint normalization stats missing. Recomputing from training years.")
        train_probe = EnergyWeatherDataset(
            weather_dir=weather_dir,
            energy_dir=energy_dir,
            years=train_years,
            selected_vars=SELECTED_VARS,
            norm_mean=None,
            norm_std=None,
            seq_len_hist=args.seq_len_hist,
            seq_len_fut=args.seq_len_fut,
            manifest_dir=manifest_dir,
            use_manifest=use_manifest,
            rebuild_manifest=rebuild_manifest,
            verbose=True,
        )
        train_targets_df = pd.DataFrame(train_probe._targets_raw.numpy(), columns=train_probe.zone_names)
        norm_mean, norm_std = compute_normalization_stats(train_targets_df)

    eval_dataset = EnergyWeatherDataset(
        weather_dir=weather_dir,
        energy_dir=energy_dir,
        years=eval_years,
        selected_vars=SELECTED_VARS,
        norm_mean=norm_mean,
        norm_std=norm_std,
        seq_len_hist=args.seq_len_hist,
        seq_len_fut=args.seq_len_fut,
        manifest_dir=manifest_dir,
        use_manifest=use_manifest,
        rebuild_manifest=rebuild_manifest,
        verbose=True,
    )
    if len(eval_dataset) == 0:
        raise RuntimeError("Evaluation dataset is empty after validity filtering.")

    sample_weather_seq = eval_dataset[0][0]
    actual_in_channels = int(sample_weather_seq.shape[1])
    metadata["in_channels"] = actual_in_channels
    metadata["num_zones"] = int(eval_dataset.num_zones)
    metadata["seq_len_hist"] = int(args.seq_len_hist)
    metadata["seq_len_fut"] = int(args.seq_len_fut)

    model = get_model(metadata).to(DEVICE)
    model.eval()

    selected_eval_indices = choose_eval_indices(
        eval_dataset,
        start_date=args.start_date,
        end_date=args.end_date,
        max_windows=args.max_windows,
        sample_frac=args.sample_frac,
        sample_seed=args.sample_seed,
        sample_mode=args.sample_mode,
    )
    eval_view, source_dataset_indices = build_eval_subset(eval_dataset, selected_eval_indices)
    print(
        f"[*] Evaluating {len(source_dataset_indices)} / {len(eval_dataset)} windows "
        f"(sample_mode={args.sample_mode}, start_date={args.start_date or 'full'}, end_date={args.end_date or 'full'})."
    )

    loader = DataLoader(
        eval_view,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    point_df = collect_point_dataframe(
        model,
        eval_dataset,
        loader,
        norm_mean,
        norm_std,
        amp_enabled,
        source_dataset_indices=source_dataset_indices,
    )

    report_tag = sanitize_name(args.model_name or (ckpt_path.stem if ckpt_path is not None else "eval"))
    years_tag = "_".join(map(str, eval_years))
    subset_tag_parts = []
    if args.start_date:
        subset_tag_parts.append(f"from-{sanitize_name(args.start_date)}")
    if args.end_date:
        subset_tag_parts.append(f"to-{sanitize_name(args.end_date)}")
    if args.max_windows > 0:
        subset_tag_parts.append(f"n-{int(args.max_windows)}")
    elif args.sample_frac < 1.0:
        subset_tag_parts.append(f"frac-{str(args.sample_frac).replace('.', 'p')}")
    subset_tag_parts.append(args.sample_mode)
    subset_tag = "__" + "__".join(subset_tag_parts) if subset_tag_parts else ""
    report_dir = Path(args.report_dir) if args.report_dir else (DEFAULT_REPORT_ROOT / f"{report_tag}__years-{years_tag}{subset_tag}")
    report_dir.mkdir(parents=True, exist_ok=True)

    save_reports(point_df, report_dir)

    if args.save_point_records:
        point_df.to_csv(report_dir / "point_level_predictions.csv", index=False)

    summary = summarize_arrays(point_df["actual"].to_numpy(), point_df["pred"].to_numpy(), point_df["baseline"].to_numpy())
    print("\n=== Evaluation summary ===")
    print(f"Windows evaluated: {len(source_dataset_indices)} / {len(eval_dataset)}")
    print(f"Points evaluated: {int(summary['count'])}")
    print(f"MAPE:           {summary['mape']:.4f}%")
    print(f"Baseline MAPE:  {summary['baseline_mape']:.4f}%")
    print(f"MAE:            {summary['mae']:.4f}")
    print(f"RMSE:           {summary['rmse']:.4f}")
    print(f"WAPE:           {summary['wape']:.4f}%")
    print(f"Bias:           {summary['bias']:.4f}")
    print(f"R^2:            {summary['r2']:.6f}")
    print(f"Calibration slope: {summary['calibration_slope']:.6f}")
    print(f"Report directory: {report_dir}")


if __name__ == "__main__":
    main()
