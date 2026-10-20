# Energy Demand Forecasting for the New England Grid

A hybrid CNN–Transformer that forecasts the next **24 hours of electricity demand** for all **8 ISO New England load zones**. Its inputs are gridded weather data, recent demand, and calendar features.

The final model reaches **3.31% MAPE** on 2023 validation data, about half the error of the course baseline (6.57%). The biggest gains came from changing *what the model predicts*: a bounded correction to a strong seasonal anchor instead of the raw load.

> **Course project:** CS 137 Deep Neural Networks, Tufts University, Spring 2026.
> Team: Odalys Garcia-Lopez, Matthew Sheng, Logan Yuan. See [Contributions](#contributions).

---

## Task

| | |
|---|---|
| **Inputs** | Hourly weather grids (450 × 449 × 7 channels) for past and future hours; past hourly demand per zone; calendar features (hour, day of week) |
| **Output** | Demand for the next 24 hours × 8 zones (CT, ME, NEMA_BOST, NH, RI, SEMA, VT, WCMA), in MWh |
| **Metric** | MAPE, averaged over all forecast hours and zones |
| **Split** | Train 2019–2022 · Validate 2023 · 2024 held out |

---

## Results

All models are compared on the same 2023 validation windows.

| Model | Val MAPE | Relative gain vs baseline | Key idea |
|---|---|---|---|
| Previous-day persistence | 6.85% | — | Copy yesterday's load at the same hour |
| Baseline CNN–Transformer | 6.57% | 0% | Predict load directly from weather + history |
| v2 | 3.79% | 42% | Gated residual on top of a previous-day baseline |
| v3 | 3.98% | 39% | v2 + specialized modules (negative result) |
| **v4 (final)** | **3.31%** | **50%** | Multiplicative correction + system-total / zone-share path |

v4 beats v2 on 68% of validation windows and improves all 8 zones.

---

## How the architecture evolved

**Baseline.** A CNN downsamples each weather frame to a 10×10 grid of tokens, and a 3-layer, 4-head Transformer encodes 24 past and 24 future steps to predict load directly. It barely beat simply copying yesterday's load, because it spent its capacity relearning the daily cycle. Attention memory also limited training to a batch size of 2.

**v2: predict the correction, not the load.** The model predicts a gated residual on top of the previous-day load:
`ŷ = baseline + sigmoid(gate) × correction`.
Separate CNNs encode past and future weather into a 4×4 tile grid, and each zone attends over those tiles. This single reframing cut MAPE by 42%.

**v3: more modules, no consistent gain.** v3 added later-horizon refinement, ramp-hour (15–19 UTC) emphasis, calendar harmonics, per-zone adapters, and difficulty-weighted losses. It beat v2 on only 39.5% of windows, and ablations clustered within 3.39–3.48%. Plausible fixes for diagnosed errors did not add up.

**v4: change the formulation again.** Detailed below.

---

## Final architecture (v4)

`models/model_adjusted_v4.py` — `SeasonalResidualHybridForecastModel`, about 1.29M parameters.

```
Weather grids ──► Shared weather CNN (16 tiles/hour) + GRUs ─┐
              └─► Weather summary encoder ───────────────────┤
Calendar ───────► Calendar encoder (hour × day-of-week) ─────┼─► Future-context Transformer
Demand history ─► Demand-history encoder ────────────────────┘   + zone-to-tile attention
              └─► Seasonal anchor (24h / 168h / 336h lags)            │
                          │                                           │
                          ├──► Zone path:   anchor × exp(0.35·tanh(ℓ)) ◄┤
                          └──► System path: total × softmax shares   ◄─┘
                                        │
                              Gated blend ─► 24 h × 8 zones
```

**Encoders**
- **Weather CNN, shared by past and future frames,** so both live in the same embedding space. Only the last 24 history hours pass through the CNN, which keeps compute bounded.
- **Weather summary encoder.** Per channel: spatial mean, std, min and max of the forecast window, plus the change from the last 6h and 24h of history. This keeps unusual values from being averaged away.
- **Calendar encoder** with an hour × day-of-week embedding, so weekday and weekend peaks are separated.
- **Demand-history encoder.** Level, trend, spread and system-level summaries of the input history.
- **Future-context Transformer** (2 layers, 4 heads) over the 24 forecast hours. Each zone also attends over the 16 weather tiles.

**Seasonal anchor.** A learned blend of the same hour 1, 7 and 14 days back. Any lag longer than the input history is masked out.

**Two prediction paths, blended per hour**
- **Zone path.** A multiplicative correction, bounded between about −30% and +42% of the anchor. It scales with each zone's load, which suits a percentage metric.
- **System + share path.** Predicts total regional load, then splits it across zones with softmax shares that sum to 1, keeping zone forecasts consistent.
- Every head's final layer starts at zero, so training begins by reproducing the anchor exactly and learns only corrections.

**Training.**
- Loss: 0.7 × Huber + 0.3 × MAPE, with mild horizon and difficulty weighting.
- Optimizer: AdamW with a cosine schedule and mixed precision. The saved weights are an exponential moving average (decay 0.999).
- Best epoch: 5 of 18.

> **Note on history length.** The final checkpoint was trained with **72 hours** of input history. The weekly and biweekly lags therefore stay masked, and the anchor reduces to the previous day. v4's gain over v2 comes from the multiplicative correction, the system/share path, the weather summaries, the calendar embedding and the longer demand history. Training with 168h or 336h of history, to switch the weekly lags on, is a natural next experiment.

---

## Error analysis: extreme weather (Part 3)

*Led by Odalys Garcia-Lopez.* Code in `error_analysis/`.

**Question:** where does the model break down, and does extreme weather drive it?

**Method:** a window counts as extreme when 2 or more weather channels fall outside their 5th–95th percentile range over its 24 forecast hours. v2 and v4 were evaluated on the same 2023 windows. The v2 checkpoint here (3.46% overall) is an earlier run than the 3.79% retrain in the results table.

| Model | Non-extreme MAPE | Extreme MAPE |
|---|---|---|
| v2 | 3.29% | 3.74% |
| v4 | 3.16% | 3.60% |

**Key findings**
- v2's error was about 14% higher in extreme weather, and the gap appears from the very first forecast hour.
- v2 had a consistent zone bias: it over-predicted CT, ME and VT, and under-predicted Boston and SEMA.
- v4, designed partly in response, improved both regimes and shifted toward slight over-prediction in nearly every zone.

| | v2 | v4 |
|---|---|---|
| MAPE, extreme vs non-extreme | ![v2 MAPE extreme vs non-extreme](folder/figures_v2/mape_extreme_vs_normal.png) | ![v4 MAPE extreme vs non-extreme](folder/figures_v4/mape_extreme_vs_normal.png) |
| Bias by zone | ![v2 bias by zone](folder/figures_v2/bias_by_zone.png) | ![v4 bias by zone](folder/figures_v4/bias_by_zone.png) |

Full analysis, including results by horizon, hour of day and weather combination, and limitations: [project summary](energy_demand_prediction_project_summary.pdf).

---

## Repository structure

```
data/
  generate_dataset.py              Dataset, sliding windows, validity manifests
  build_validity_manifest.py       Prebuild the manifests before training
models/
  model_adjusted_v2.py             v2: gated additive residual
  model_adjusted_v3.py             v3: v2 + specialized modules
  model_adjusted_v4.py             v4: final model
training/
  train_adjusted_ema_accum_v2.py   Trains v2
  train_adjusted_v3.py             Trains v3 / v4 (loads model_adjusted_v4 first)
  smoke_test_adjusted_ema_accum_v2.py  Tiny end-to-end check
evaluation/
  evaluate_diagnostics_v2.py       Metrics by zone, horizon, hour, month, peak slices
  analyze_dataset.py               Data profiling and no-retraining ablations
error_analysis/
  extract_energy_errors.py         Inference → predictions.csv, weather_stats.csv
  analyze_extreme_errors.py        Extreme-weather flags, summary table, figures
  extract_errs.sh, analysis_errs.sh  SLURM job scripts
folder/
  figures_v2/                      Error-analysis figures for v2
  figures_v4/                      Error-analysis figures for v4
energy_demand_prediction_project_summary.pdf   Project summary
```

---

## Data

The data was provided through the course on the Tufts HPC cluster and is **not included or redistributable**. It consists of:
- **Weather:** one tensor per hour, `weather_data/<year>/X_<YYYYMMDDHH>.pt`, of shape (450, 449, 7).
- **Demand:** `energy_demand_data/target_energy_zonal_<year>.csv`, hourly MWh for the 8 zones, timestamps in UTC.

Because the data and the trained checkpoints stay on the cluster, the scripts in this repo can't be run without access to it.

---

## How the pipeline runs

1. **Build validity manifests** (`data/build_validity_manifest.py`): scan the weather files once and cache which hourly windows are usable, so training doesn't rescan them.
2. **Train** (`training/train_adjusted_v3.py`, which loads the v4 model): saves the best checkpoint by validation MAPE.
3. **Error analysis**: `error_analysis/extract_energy_errors.py` runs the trained model on 2023 and saves per-point errors and per-window weather stats. Then `error_analysis/analyze_extreme_errors.py` flags extreme-weather windows and produces the tables and figures.


```bash
python training/train_adjusted_v3.py --model-name seasonal_v4 --seq-len-hist 72 \
  --batch-size 2 --grad-accum-steps 2 --epochs 18 --zone-adapter-rank 0 \
  --later-horizon-strength 0.0 --horizon-loss-max-weight 1.05 \
  --difficulty-weight-alpha 0.1 --difficulty-weight-cap 2.0 \
  --ramp-weight-alpha 0.0 --huber-mix 0.7 --mape-mix 0.3 --skip-nonfinite-batches
```

---

## Contributions

- **Odalys Garcia-Lopez:** Part 3 error analysis. Built the extreme-weather analysis pipeline (`error_analysis/`) and analyzed v2 failure modes and how v4 addressed them.
- **Matthew Sheng:** Valid-file manifests and data loading, data exploration, model architectures v2–v4 and ablations, training and comparison scripts.
- **Logan Yuan:** Baseline model (Part 1); graphical analysis of v4 in Part 3.
