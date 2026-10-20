"""
analyze_extreme_errors.py
=========================
Loads the two CSVs produced by extract_extreme_errors.py and performs a
comprehensive error analysis, broken down by:

  • Extreme vs. non-extreme windows (per-channel percentile thresholds)
  • Load zone
  • Forecast horizon (hours 1-24)
  • Extreme-weather type (one plot per channel that was extreme)

Outputs (all inside --output-dir):
  extreme_windows.csv          which windows were flagged and why
  summary_table.csv            MAPE by zone × extreme/non-extreme
  figures/mape_extreme_vs_normal.png
  figures/mape_by_zone.png
  figures/mape_by_horizon.png
  figures/mape_by_channel_type.png
  figures/error_distribution.png

Usage:
    python analyze_extreme_errors.py \
        --output-dir ./extreme_analysis \
        --extreme-pct 5          # flag top/bottom 5th-percentile per channel
        --min-channels-hit 1     # flag window if ≥ 1 channel is extreme
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch


# ── Plotting helpers ──────────────────────────────────────────────────────────

PALETTE = {
    "extreme"     : "#d62728",
    "non_extreme" : "#1f77b4",
    "neutral"     : "#7f7f7f",
    "highlight"   : "#ff7f0e",
}


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ── Core analysis functions ───────────────────────────────────────────────────

def flag_extreme_windows(
    wx_df: pd.DataFrame,
    channel_names: list,
    extreme_pct: float,
    min_channels_hit: int,
) -> pd.DataFrame:
    """
    For each channel, flag windows whose future spatial mean falls in the
    top or bottom `extreme_pct` percentile across all windows.

    Returns wx_df with extra columns:
        ch_<name>_extreme   : bool
        ch_<name>_direction : "high" | "low" | ""
        is_extreme          : bool (True if ≥ min_channels_hit channels flagged)
        extreme_channels    : comma-separated list of flagged channel names
        extreme_types       : e.g.  "ch0_high,ch2_low"
    """
    df = wx_df.copy()
    lo = extreme_pct / 100.0
    hi = 1.0 - lo

    flagged_cols   = []
    direction_cols = []

    for cname in channel_names:
        safe    = cname.replace(" ", "_")
        col     = f"{safe}_fut_max"
        if col not in df.columns:
            continue

        q_lo = df[col].quantile(lo)
        q_hi = df[col].quantile(hi)

        is_low  = df[col] <= q_lo
        is_high = df[col] >= q_hi

        flag_col = f"ch_{safe}_extreme"
        dir_col  = f"ch_{safe}_direction"
        df[flag_col] = is_low | is_high
        df[dir_col]  = np.where(is_high, "high", np.where(is_low, "low", ""))
        flagged_cols.append(flag_col)
        direction_cols.append((safe, dir_col))

    # Aggregate across channels
    df["num_channels_extreme"] = df[flagged_cols].sum(axis=1)
    df["is_extreme"] = df["num_channels_extreme"] >= min_channels_hit

    def _build_type_str(row):
        parts = []
        for safe, dc in direction_cols:
            d = row[dc]
            if d:
                parts.append(f"{safe}_{d}")
        return ",".join(parts)

    df["extreme_channels"] = df[[f"ch_{s}_extreme" for s, _ in direction_cols]].apply(
        lambda row: ",".join(s for s, _ in direction_cols if row[f"ch_{s}_extreme"]), axis=1
    )
    df["extreme_types"] = df.apply(_build_type_str, axis=1)
    return df


def merge_labels(pred_df: pd.DataFrame, wx_df: pd.DataFrame) -> pd.DataFrame:
    """Merge is_extreme and extreme_types from wx_df into pred_df."""
    labels = wx_df[["window_start_utc", "is_extreme", "extreme_types", "num_channels_extreme"]].copy()
    pred_df = pred_df.merge(labels, on="window_start_utc", how="left")
    pred_df["is_extreme"] = pred_df["is_extreme"].fillna(False)
    return pred_df


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_mape_extreme_vs_normal(pred_df: pd.DataFrame, fig_dir: Path):
    """Bar chart: overall MAPE for extreme vs non-extreme windows."""
    grp  = pred_df.groupby("is_extreme")["pct_error"].mean()
    cats = ["Non-Extreme", "Extreme"]
    vals = [grp.get(False, np.nan), grp.get(True, np.nan)]
    cols = [PALETTE["non_extreme"], PALETTE["extreme"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(cats, vals, color=cols, width=0.5, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Absolute Percentage Error (%)", fontsize=11)
    ax.set_title("Model MAPE: Extreme vs. Non-Extreme Weather Windows", fontsize=12, pad=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.25)
    fig.tight_layout()
    _save(fig, fig_dir / "mape_extreme_vs_normal.png")


def plot_mape_by_zone(pred_df: pd.DataFrame, fig_dir: Path):
    """Grouped bar chart: MAPE per zone × extreme/non-extreme."""
    grp  = pred_df.groupby(["zone", "is_extreme"])["pct_error"].mean().unstack(fill_value=np.nan)
    zones = grp.index.tolist()
    x     = np.arange(len(zones))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(zones) * 1.1), 4.5))
    if False in grp.columns:
        ax.bar(x - width / 2, grp[False], width, label="Non-Extreme",
               color=PALETTE["non_extreme"], edgecolor="white")
    if True in grp.columns:
        ax.bar(x + width / 2, grp[True],  width, label="Extreme",
               color=PALETTE["extreme"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(zones, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("MAPE (%)", fontsize=11)
    ax.set_title("MAPE by Load Zone: Extreme vs. Non-Extreme Windows", fontsize=12, pad=10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, fig_dir / "mape_by_zone.png")


def plot_mape_by_horizon(pred_df: pd.DataFrame, fig_dir: Path):
    """Line plot: MAPE vs. forecast hour, split by extreme/non-extreme."""
    grp = pred_df.groupby(["forecast_hour", "is_extreme"])["pct_error"].mean().unstack(fill_value=np.nan)
    hours = grp.index.tolist()

    fig, ax = plt.subplots(figsize=(9, 4))
    if False in grp.columns:
        ax.plot(hours, grp[False], color=PALETTE["non_extreme"], lw=2.5,
                marker="o", ms=4, label="Non-Extreme")
    if True in grp.columns:
        ax.plot(hours, grp[True], color=PALETTE["extreme"], lw=2.5,
                marker="o", ms=4, label="Extreme")
    ax.set_xlabel("Forecast Hour (1 = next hour)", fontsize=11)
    ax.set_ylabel("MAPE (%)", fontsize=11)
    ax.set_title("MAPE vs. Forecast Horizon: Extreme vs. Non-Extreme Windows", fontsize=12, pad=10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(hours[::2])
    fig.tight_layout()
    _save(fig, fig_dir / "mape_by_horizon.png")


def plot_mape_by_channel_type(pred_df: pd.DataFrame, fig_dir: Path, top_n: int = 8):
    """
    Bar chart: MAPE grouped by which channels were extreme.
    Only shows the top_n most common extreme types.
    """
    # Rows that belong to extreme windows
    ext = pred_df[pred_df["is_extreme"]].copy()
    if ext.empty:
        print("  [skip] No extreme windows found for channel-type plot.")
        return

    grp = ext.groupby("extreme_types")["pct_error"].mean().sort_values(ascending=False)
    grp = grp.head(top_n)
    baseline = pred_df[~pred_df["is_extreme"]]["pct_error"].mean()

    fig, ax = plt.subplots(figsize=(max(6, len(grp) * 1.0), 4.5))
    colors = [PALETTE["extreme"]] * len(grp)
    ax.barh(range(len(grp)), grp.values, color=colors, edgecolor="white")
    ax.axvline(baseline, color=PALETTE["non_extreme"], lw=2, ls="--",
               label=f"Non-extreme baseline ({baseline:.2f}%)")
    ax.set_yticks(range(len(grp)))
    ax.set_yticklabels(grp.index.tolist(), fontsize=9)
    ax.set_xlabel("Mean MAPE (%)", fontsize=11)
    ax.set_title("MAPE by Extreme-Weather Type\n(top combinations, extreme windows only)",
                 fontsize=12, pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, fig_dir / "mape_by_channel_type.png")


def plot_error_distribution(pred_df: pd.DataFrame, fig_dir: Path):
    """KDE/histogram of signed errors for extreme vs non-extreme."""
    ext     = pred_df[pred_df["is_extreme"]]["signed_error_mwh"].values
    non_ext = pred_df[~pred_df["is_extreme"]]["signed_error_mwh"].values

    # Clip for readability (5th–95th percentile)
    clip_lo = np.percentile(pred_df["signed_error_mwh"], 2)
    clip_hi = np.percentile(pred_df["signed_error_mwh"], 98)
    bins    = np.linspace(clip_lo, clip_hi, 60)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(non_ext, bins=bins, density=True, alpha=0.6, color=PALETTE["non_extreme"],
            label="Non-Extreme", edgecolor="none")
    ax.hist(ext,     bins=bins, density=True, alpha=0.65, color=PALETTE["extreme"],
            label="Extreme", edgecolor="none")
    ax.axvline(0, color=PALETTE["neutral"], lw=1.5, ls="--")
    ax.set_xlabel("Signed Error (pred − true, MWh)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Signed Error Distribution: Extreme vs. Non-Extreme Windows",
                 fontsize=12, pad=10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, fig_dir / "error_distribution.png")


def plot_mape_by_hour_of_day(pred_df: pd.DataFrame, fig_dir: Path):
    """
    Line chart: MAPE by actual hour-of-day (UTC) rather than forecast step.
    Helps reveal whether extreme-weather errors cluster around peak-demand hours.
    """
    df = pred_df.copy()
    df["hour_of_day"] = pd.to_datetime(df["forecast_utc"]).dt.hour
    grp = df.groupby(["hour_of_day", "is_extreme"])["pct_error"].mean().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(9, 4))
    hours = list(range(24))
    if False in grp.columns:
        ax.plot(hours, [grp[False].get(h, np.nan) for h in hours],
                color=PALETTE["non_extreme"], lw=2.5, marker="o", ms=4, label="Non-Extreme")
    if True in grp.columns:
        ax.plot(hours, [grp[True].get(h, np.nan) for h in hours],
                color=PALETTE["extreme"], lw=2.5, marker="o", ms=4, label="Extreme")
    ax.set_xlabel("Hour of Day (UTC)", fontsize=11)
    ax.set_ylabel("MAPE (%)", fontsize=11)
    ax.set_title("MAPE by Hour-of-Day: Extreme vs. Non-Extreme Windows", fontsize=12, pad=10)
    ax.legend(frameon=False)
    ax.set_xticks(range(0, 24, 2))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, fig_dir / "mape_by_hour_of_day.png")


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    MAPE, MAE, bias by zone × {non-extreme, extreme} with count of windows.
    """
    rows = []
    for is_ext in [False, True]:
        label = "Extreme" if is_ext else "Non-Extreme"
        sub   = pred_df[pred_df["is_extreme"] == is_ext]
        for zone in sorted(pred_df["zone"].unique()):
            z = sub[sub["zone"] == zone]
            rows.append({
                "Weather Condition"  : label,
                "Zone"               : zone,
                "N Windows"          : z["window_start_utc"].nunique(),
                "N Predictions"      : len(z),
                "Mean MAPE (%)"      : round(z["pct_error"].mean(), 3),
                "Median MAPE (%)"    : round(z["pct_error"].median(), 3),
                "MAE (MWh)"          : round(z["abs_error_mwh"].mean(), 1),
                "Mean Bias (MWh)"    : round(z["signed_error_mwh"].mean(), 1),
                "Std Error (MWh)"    : round(z["signed_error_mwh"].std(), 1),
            })
    # Add "All Zones" aggregate
    for is_ext in [False, True]:
        label = "Extreme" if is_ext else "Non-Extreme"
        sub   = pred_df[pred_df["is_extreme"] == is_ext]
        rows.append({
            "Weather Condition"  : label,
            "Zone"               : "ALL",
            "N Windows"          : sub["window_start_utc"].nunique(),
            "N Predictions"      : len(sub),
            "Mean MAPE (%)"      : round(sub["pct_error"].mean(), 3),
            "Median MAPE (%)"    : round(sub["pct_error"].median(), 3),
            "MAE (MWh)"          : round(sub["abs_error_mwh"].mean(), 1),
            "RMSE (MWh)": round(np.sqrt((sub["signed_error_mwh"] ** 2).mean()), 1),
            "Mean Bias (MWh)"    : round(sub["signed_error_mwh"].mean(), 1),
            "Std Error (MWh)"    : round(sub["signed_error_mwh"].std(), 1),
        })
    return pd.DataFrame(rows)

def plot_bias_by_zone(pred_df: pd.DataFrame, fig_dir: Path):
    """Mean signed error (bias) per zone × extreme/non-extreme."""
    grp  = pred_df.groupby(["zone", "is_extreme"])["signed_error_mwh"].mean().unstack(fill_value=np.nan)
    zones = grp.index.tolist()
    x     = np.arange(len(zones))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(zones) * 1.1), 4.5))
    if False in grp.columns:
        ax.bar(x - width / 2, grp[False], width, label="Non-Extreme",
               color=PALETTE["non_extreme"], edgecolor="white")
    if True in grp.columns:
        ax.bar(x + width / 2, grp[True], width, label="Extreme",
               color=PALETTE["extreme"], edgecolor="white")
    ax.axhline(0, color=PALETTE["neutral"], lw=1.2, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(zones, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Mean Signed Error (pred − true, MWh)", fontsize=11)
    ax.set_title("Forecast Bias by Load Zone: Extreme vs. Non-Extreme Windows",
                 fontsize=12, pad=10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, fig_dir / "bias_by_zone.png")

# ── CLI & main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Analyse extreme-weather model errors.")
    p.add_argument("--output-dir",        default="./extreme_analysis",
                   help="Same folder used in extract_extreme_errors.py.")
    p.add_argument("--extreme-pct",       type=float, default=5.0,
                   help="Percentile threshold (each tail). Default: 5 = top/bottom 5%%.")
    p.add_argument("--min-channels-hit",  type=int,   default=1,
                   help="Min channels that must be extreme to flag window. Default: 1.")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pred_csv  = out_dir / "predictions.csv"
    wx_csv    = out_dir / "weather_stats.csv"
    meta_path = out_dir / "extraction_meta.pt"

    for f in [pred_csv, wx_csv]:
        if not f.exists():
            raise FileNotFoundError(
                f"Expected {f}. Run extract_extreme_errors.py first."
            )

    print("[*] Loading CSVs …")
    pred_df = pd.read_csv(pred_csv,  parse_dates=["window_start_utc", "forecast_utc"])
    wx_df   = pd.read_csv(wx_csv,    parse_dates=["window_start_utc"])

    # Load channel names from metadata if available
    channel_names = None
    if meta_path.exists():
        meta          = torch.load(meta_path, map_location="cpu", weights_only=False)
        channel_names = meta.get("channel_names", None)
    if channel_names is None:
        # Infer from weather_stats columns
        channel_names = sorted({
            c.replace("_fut_mean", "").replace("_fut_max", "")
            for c in wx_df.columns
            if c.endswith("_fut_mean")
        })
    print(f"[*] Channels detected : {channel_names}")
    print(f"[*] Total windows     : {len(wx_df):,}")
    print(f"[*] Total predictions : {len(pred_df):,}")

    # ── Flag extreme windows ──────────────────────────────────────────────────
    print(f"\n[*] Flagging extreme windows "
          f"(top/bottom {args.extreme_pct:.0f}th percentile, "
          f"≥{args.min_channels_hit} channel(s)) …")
    wx_df = flag_extreme_windows(
        wx_df,
        channel_names,
        extreme_pct        = args.extreme_pct,
        min_channels_hit   = args.min_channels_hit,
    )
    n_ext     = wx_df["is_extreme"].sum()
    n_non_ext = (~wx_df["is_extreme"]).sum()
    print(f"    Extreme windows     : {n_ext:,}  ({100*n_ext/len(wx_df):.1f}%)")
    print(f"    Non-extreme windows : {n_non_ext:,}  ({100*n_non_ext/len(wx_df):.1f}%)")

    # Merge labels into predictions
    pred_df = merge_labels(pred_df, wx_df)

    # ── Save flagged windows ──────────────────────────────────────────────────
    ext_windows_path = out_dir / "extreme_windows.csv"
    wx_df.to_csv(ext_windows_path, index=False)
    print(f"[✓] Extreme window flags → {ext_windows_path}")

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\n[*] Building summary table …")
    summary = build_summary_table(pred_df)
    summary_path = out_dir / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[✓] Summary table → {summary_path}")

    # Print the ALL-zones rows to console
    print("\n── Quick Summary (All Zones) ──────────────────────────────────────────")
    print(summary[summary["Zone"] == "ALL"].to_string(index=False))

    # Per-channel breakdown
    print("\n── Extreme Windows by Channel ─────────────────────────────────────────")
    for cname in channel_names:
        safe = cname.replace(" ", "_")
        col  = f"ch_{safe}_extreme"
        if col in wx_df.columns:
            n = wx_df[col].sum()
            pct = 100 * n / len(wx_df)
            print(f"  {cname:<30s}: {n:4d} windows ({pct:.1f}%)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[*] Generating figures …")
    plot_mape_extreme_vs_normal(pred_df, fig_dir)
    plot_mape_by_zone(pred_df, fig_dir)
    plot_mape_by_horizon(pred_df, fig_dir)
    plot_mape_by_channel_type(pred_df, fig_dir)
    plot_error_distribution(pred_df, fig_dir)
    plot_mape_by_hour_of_day(pred_df, fig_dir)
    plot_bias_by_zone(pred_df, fig_dir)

    print(f"\n[✓] All figures saved to {fig_dir}/")
    print("\nDone!  Key files:")
    print(f"  {summary_path}")
    print(f"  {ext_windows_path}")
    print(f"  {fig_dir}/")


if __name__ == "__main__":
    main()