# V4 Latest model architecture

"""
model_adjusted_v4.py

Drop-in model replacement for the current train_adjusted_v3-style training script.

Key compatibility points:
- Exposes get_model(metadata: Dict[str, object]).
- Forward signature is exactly:
      model(weather_seq, hist_demand, hist_cal, fut_cal) -> [B, F, Z]
- Supports set_weather_stats(...) and set_target_stats(...).
- Accepts the same metadata/config keys used by the current train file, including
  later_horizon_*, ramp_*, hour_band_size, use_calendar_second_harmonics, and
  zone_adapter_rank. Some of these are retained for checkpoint/config compatibility.

Key modeling changes versus the previous residual module stack:
- Stronger seasonal baseline: daily + weekly + biweekly aligned lags when available.
- Demand-only long-history support: if seq_len_hist is 168/336, only the last
  weather_history_steps weather frames go through the CNN, while all load history
  is used by cheap tabular/lag encoders.
- MAPE-aligned multiplicative correction: predicts bounded log-ratio corrections
  around the seasonal baseline and returns normalized predictions to the existing
  training loss.
- Cheap future weather summaries: channel-wise map statistics and future-minus-
  historical summary deltas are fused with CNN weather embeddings.
- System-load + zone-share consistency path: combines a direct zone correction
  with a coherent total-load/share correction, both initialized to exactly reproduce
  the baseline at the start of training.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================
# Small utility helpers
# ============================================================


def _as_tuple2(value, default: Tuple[int, int] = (4, 4)) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = value.replace("x", ",").replace(" ", "").split(",")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
        return default
    if isinstance(value, Iterable):
        value = tuple(value)
        if len(value) == 2:
            return int(value[0]), int(value[1])
    return default


def _cyclical_to_index(cal_feats: torch.Tensor, sin_idx: int, cos_idx: int, period: int) -> torch.Tensor:
    """Recover integer hour/day index from sin/cos calendar features."""
    angle = torch.atan2(cal_feats[..., sin_idx], cal_feats[..., cos_idx])
    frac = torch.remainder(angle / (2.0 * math.pi), 1.0)
    idx = torch.round(frac * float(period)).long() % period
    return idx


def _hours_in_range(hour_idx: torch.Tensor, start_hour: int, end_hour: int) -> torch.Tensor:
    start_hour = int(start_hour) % 24
    end_hour = int(end_hour) % 24
    if start_hour <= end_hour:
        return (hour_idx >= start_hour) & (hour_idx <= end_hour)
    return (hour_idx >= start_hour) | (hour_idx <= end_hour)


def _aligned_lag(hist_demand: torch.Tensor, lag: int, fut_len: int) -> Optional[torch.Tensor]:
    """
    For each future horizon h=1..F, return y[t+h-lag].

    hist_demand[:, -1] is y[t]. Therefore for lag=24 and F=24 this returns
    hist_demand[:, -24:] == y[t-23], ..., y[t], matching t+1..t+24 to the
    previous day's same hour.
    """
    steps = int(hist_demand.size(1))
    if steps < int(lag):
        return None
    start = steps - int(lag)
    end = start + int(fut_len)
    if start < 0 or end > steps:
        return None
    return hist_demand[:, start:end, :]


def _last_window(hist_demand: torch.Tensor, fut_len: int) -> torch.Tensor:
    """Previous-day fallback used when only a short 24h history exists."""
    if hist_demand.size(1) >= fut_len:
        return hist_demand[:, -fut_len:, :]
    pad = hist_demand[:, -1:, :].expand(-1, fut_len - hist_demand.size(1), -1)
    return torch.cat([pad, hist_demand], dim=1)[:, -fut_len:, :]


def _safe_std(x: torch.Tensor, dim, keepdim: bool = False) -> torch.Tensor:
    if x.size(dim if isinstance(dim, int) else dim[0]) <= 1:
        return torch.zeros_like(x.mean(dim=dim, keepdim=keepdim))
    return x.std(dim=dim, unbiased=False, keepdim=keepdim)


# ============================================================
# Encoders
# ============================================================


class WeatherChannelGate(nn.Module):
    """A neutral-initialized channel gate. Starts as identity."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        hidden = max(8, min(64, int(in_channels) * 4))
        self.norm = nn.LayerNorm(int(in_channels))
        self.fc1 = nn.Linear(int(in_channels), hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, int(in_channels))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = x.mean(dim=(2, 3))
        delta = 0.35 * torch.tanh(self.fc2(self.act(self.fc1(self.norm(stats)))))
        return x * (1.0 + delta).unsqueeze(-1).unsqueeze(-1)


class WeatherSpatialEncoder(nn.Module):
    """
    Light CNN that produces both spatial patch tokens and a global frame embedding.

    This is intentionally close to the control architecture so runtime stays stable.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        grid_size: Tuple[int, int] = (4, 4),
        hidden_channels: int = 64,
        dropout: float = 0.1,
        force_fp32: bool = True,
    ) -> None:
        super().__init__()
        self.force_fp32 = bool(force_fp32)
        self.channel_gate = WeatherChannelGate(in_channels)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(grid_size),
        )
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_channels, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def _forward_impl(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.channel_gate(x)
        feat = self.backbone(x)
        tokens = feat.flatten(2).transpose(1, 2)  # [B, P, C]
        token_emb = self.token_proj(tokens)
        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.amax(dim=1)
        global_emb = self.global_proj(torch.cat([mean_pool, max_pool], dim=-1))
        return token_emb, global_emb

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.force_fp32 and x.is_floating_point():
            with torch.autocast(device_type=x.device.type, enabled=False):
                token_emb, global_emb = self._forward_impl(x.float())
            return token_emb.float(), global_emb.float()
        return self._forward_impl(x)


class WeatherSummaryEncoder(nn.Module):
    """Cheap channel-statistic weather encoder fused with the CNN path."""

    def __init__(self, in_channels: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        # future mean/std/min/max + delta to recent historical mean + delta to daily historical mean
        self.raw_dim = 6 * self.in_channels
        hidden = max(out_dim, 64)
        self.proj = nn.Sequential(
            nn.Linear(self.raw_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    @staticmethod
    def _frame_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, C, H, W]
        mean = x.mean(dim=(-2, -1))
        var = (x - mean.unsqueeze(-1).unsqueeze(-1)).pow(2).mean(dim=(-2, -1))
        std = torch.sqrt(var.clamp_min(1e-6))
        minv = x.amin(dim=(-2, -1))
        maxv = x.amax(dim=(-2, -1))
        return mean, std, minv, maxv

    def forward(self, hist_weather: torch.Tensor, fut_weather: torch.Tensor) -> torch.Tensor:
        fut_mean, fut_std, fut_min, fut_max = self._frame_stats(fut_weather)
        hist_mean, _, _, _ = self._frame_stats(hist_weather)
        recent_w = min(6, hist_mean.size(1))
        day_w = min(24, hist_mean.size(1))
        hist_recent = hist_mean[:, -recent_w:, :].mean(dim=1, keepdim=True)
        hist_day = hist_mean[:, -day_w:, :].mean(dim=1, keepdim=True)
        delta_recent = fut_mean - hist_recent
        delta_day = fut_mean - hist_day
        raw = torch.cat([fut_mean, fut_std, fut_min, fut_max, delta_recent, delta_day], dim=-1)
        return self.proj(raw)


class CalendarFeatureEncoder(nn.Module):
    """Hour/day encoder compatible with the existing 4-dim calendar tensors."""

    def __init__(
        self,
        out_dim: int,
        raw_dim: int = 4,
        dropout: float = 0.1,
        ramp_start_hour: int = 15,
        ramp_end_hour: int = 19,
        hour_band_size: int = 4,
        use_second_harmonics: bool = True,
    ) -> None:
        super().__init__()
        emb_dim = max(8, out_dim // 10)
        self.ramp_start_hour = int(ramp_start_hour)
        self.ramp_end_hour = int(ramp_end_hour)
        self.hour_band_size = max(1, int(hour_band_size))
        self.use_second_harmonics = bool(use_second_harmonics)
        self.num_hour_bands = int(math.ceil(24 / float(self.hour_band_size)))

        self.hour_emb = nn.Embedding(24, emb_dim)
        self.dow_emb = nn.Embedding(7, emb_dim)
        self.weekend_emb = nn.Embedding(2, emb_dim)
        self.ramp_emb = nn.Embedding(2, emb_dim)
        self.hour_band_emb = nn.Embedding(self.num_hour_bands, emb_dim)
        self.hour_dow_emb = nn.Embedding(24 * 7, emb_dim)

        harmonic_dim = 4 if self.use_second_harmonics else 0
        interaction_dim = 4  # sin_h*sin_d, sin_h*cos_d, cos_h*sin_d, cos_h*cos_d
        self.raw_proj = nn.Sequential(
            nn.Linear(raw_dim + harmonic_dim + interaction_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_proj = nn.Sequential(
            nn.Linear((out_dim // 2) + 6 * emb_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, cal_feats: torch.Tensor) -> torch.Tensor:
        hour_idx = _cyclical_to_index(cal_feats, 0, 1, 24)
        dow_idx = _cyclical_to_index(cal_feats, 2, 3, 7)
        weekend_idx = (dow_idx >= 5).long()
        ramp_idx = _hours_in_range(hour_idx, self.ramp_start_hour, self.ramp_end_hour).long()
        hour_band_idx = torch.clamp(hour_idx // self.hour_band_size, max=self.num_hour_bands - 1)
        hour_dow_idx = hour_idx * 7 + dow_idx

        raw_parts = [cal_feats]
        sin_h, cos_h = cal_feats[..., 0], cal_feats[..., 1]
        sin_d, cos_d = cal_feats[..., 2], cal_feats[..., 3]
        if self.use_second_harmonics:
            raw_parts.append(
                torch.stack(
                    [
                        2.0 * sin_h * cos_h,
                        cos_h.pow(2) - sin_h.pow(2),
                        2.0 * sin_d * cos_d,
                        cos_d.pow(2) - sin_d.pow(2),
                    ],
                    dim=-1,
                )
            )
        raw_parts.append(
            torch.stack(
                [sin_h * sin_d, sin_h * cos_d, cos_h * sin_d, cos_h * cos_d],
                dim=-1,
            )
        )
        raw = self.raw_proj(torch.cat(raw_parts, dim=-1))
        emb = torch.cat(
            [
                self.hour_emb(hour_idx),
                self.dow_emb(dow_idx),
                self.weekend_emb(weekend_idx),
                self.ramp_emb(ramp_idx),
                self.hour_band_emb(hour_band_idx),
                self.hour_dow_emb(hour_dow_idx),
            ],
            dim=-1,
        )
        return self.out_proj(torch.cat([raw, emb], dim=-1))


class DemandHistoryEncoder(nn.Module):
    """Cheap long-history demand encoder. Can process 24, 168, or 336 steps."""

    def __init__(self, num_zones: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_zones = int(num_zones)
        # last, recent6, day, week, full, std, trend, daily_delta, weekly_delta = 9Z
        # plus system-level summaries = 6
        in_dim = 9 * self.num_zones + 6
        hidden = max(out_dim, 96)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, hist_demand: torch.Tensor) -> torch.Tensor:
        steps = hist_demand.size(1)
        last = hist_demand[:, -1, :]
        recent_w = min(6, steps)
        day_w = min(24, steps)
        week_w = min(168, steps)
        recent_mean = hist_demand[:, -recent_w:, :].mean(dim=1)
        day_mean = hist_demand[:, -day_w:, :].mean(dim=1)
        week_mean = hist_demand[:, -week_w:, :].mean(dim=1)
        full_mean = hist_demand.mean(dim=1)
        full_std = hist_demand.std(dim=1, unbiased=False)
        trend = last - hist_demand[:, 0, :]
        daily_delta = last - hist_demand[:, -24, :] if steps >= 24 else torch.zeros_like(last)
        weekly_delta = last - hist_demand[:, -168, :] if steps >= 168 else daily_delta

        system_last = last.mean(dim=-1, keepdim=True)
        system_day = day_mean.mean(dim=-1, keepdim=True)
        system_week = week_mean.mean(dim=-1, keepdim=True)
        system_std = full_std.mean(dim=-1, keepdim=True)
        system_trend = trend.mean(dim=-1, keepdim=True)
        zone_imbalance = (last - system_last).abs().mean(dim=-1, keepdim=True)

        feat = torch.cat(
            [
                last,
                recent_mean,
                day_mean,
                week_mean,
                full_mean,
                full_std,
                trend,
                daily_delta,
                weekly_delta,
                system_last,
                system_day,
                system_week,
                system_std,
                system_trend,
                zone_imbalance,
            ],
            dim=-1,
        )
        return self.proj(feat)


class SeasonalLagEncoder(nn.Module):
    """Builds aligned lag features and a learned masked seasonal baseline."""

    def __init__(self, num_zones: int, fut_len: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_zones = int(num_zones)
        self.fut_len = int(fut_len)
        # daily, weekly, biweekly logits. Starts mostly daily, but can learn blend.
        self.baseline_logits = nn.Parameter(torch.tensor([2.0, 0.75, 0.25], dtype=torch.float32))
        # K features below. Keep this explicit so shape bugs are easy to catch.
        self.feature_dim = 18
        hidden = max(out_dim, 64)
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def _lag_or(self, hist_demand: torch.Tensor, lag: int, fallback: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        value = _aligned_lag(hist_demand, lag, self.fut_len)
        if value is None:
            return fallback, False
        return value, True

    def forward(self, hist_demand: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, steps, zones = hist_demand.shape
        daily = _last_window(hist_demand, self.fut_len)
        weekly, has_weekly = self._lag_or(hist_demand, 168, daily)
        biweekly, has_biweekly = self._lag_or(hist_demand, 336, weekly)

        lag_stack = torch.stack([daily, weekly, biweekly], dim=-1)  # [B, F, Z, 3]
        avail = torch.tensor(
            [1.0, 1.0 if has_weekly else 0.0, 1.0 if has_biweekly else 0.0],
            device=hist_demand.device,
            dtype=hist_demand.dtype,
        )
        logits = self.baseline_logits.to(device=hist_demand.device, dtype=hist_demand.dtype)
        masked_logits = logits.masked_fill(avail <= 0, -1.0e4)
        weights = torch.softmax(masked_logits, dim=0).view(1, 1, 1, 3)
        baseline = (lag_stack * weights).sum(dim=-1)

        recent_w = min(6, steps)
        day_w = min(24, steps)
        week_w = min(168, steps)
        last = hist_demand[:, -1:, :].expand(-1, self.fut_len, -1)
        recent_mean = hist_demand[:, -recent_w:, :].mean(dim=1, keepdim=True).expand(-1, self.fut_len, -1)
        day_mean = hist_demand[:, -day_w:, :].mean(dim=1, keepdim=True).expand(-1, self.fut_len, -1)
        week_mean = hist_demand[:, -week_w:, :].mean(dim=1, keepdim=True).expand(-1, self.fut_len, -1)
        full_mean = hist_demand.mean(dim=1, keepdim=True).expand(-1, self.fut_len, -1)
        full_std = hist_demand.std(dim=1, unbiased=False, keepdim=True).expand(-1, self.fut_len, -1)

        # System-level baseline and zone-share features.
        base_sys = baseline.mean(dim=-1, keepdim=True).expand(-1, -1, zones)
        daily_sys = daily.mean(dim=-1, keepdim=True).expand(-1, -1, zones)
        weekly_sys = weekly.mean(dim=-1, keepdim=True).expand(-1, -1, zones)
        baseline_share_norm = baseline - base_sys
        daily_minus_weekly = daily - weekly
        last_minus_daily = last - daily
        baseline_minus_week = baseline - weekly

        # Horizon progress lets the MLP learn short vs late horizon behavior cheaply.
        if self.fut_len <= 1:
            h = torch.zeros(1, self.fut_len, 1, device=hist_demand.device, dtype=hist_demand.dtype)
        else:
            h = torch.linspace(0.0, 1.0, self.fut_len, device=hist_demand.device, dtype=hist_demand.dtype).view(1, self.fut_len, 1)
        h = h.expand(bsz, -1, zones)

        feats = torch.stack(
            [
                baseline,
                daily,
                weekly,
                biweekly,
                last,
                recent_mean,
                day_mean,
                week_mean,
                full_mean,
                full_std,
                daily_minus_weekly,
                last_minus_daily,
                baseline_minus_week,
                base_sys,
                daily_sys,
                weekly_sys,
                baseline_share_norm,
                h,
            ],
            dim=-1,
        )
        emb = self.proj(feats)
        return baseline, emb


# ============================================================
# Main model
# ============================================================


class SeasonalResidualHybridForecastModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_zones: int,
        seq_len_hist: int = 24,
        seq_len_fut: int = 24,
        weather_dim: int = 128,
        tab_dim: int = 128,
        decoder_dim: int = 128,
        grid_size: Tuple[int, int] = (4, 4),
        num_future_layers: int = 2,
        num_heads: int = 4,
        head_hidden: int = 128,
        dropout: float = 0.1,
        weather_norm_clip: Optional[float] = 10.0,
        weather_encoder_force_fp32: bool = True,
        # Compatibility / optional knobs from current train file.
        later_horizon_start_step: int = 8,
        later_horizon_strength: float = 0.0,
        later_horizon_power: float = 1.5,
        ramp_start_hour: int = 15,
        ramp_end_hour: int = 19,
        hour_band_size: int = 4,
        use_calendar_second_harmonics: bool = True,
        zone_adapter_rank: int = 0,
        # New knobs. They default safely when train does not pass them.
        weather_history_steps: int = 24,
        log_residual_scale: float = 0.35,
        system_log_residual_scale: float = 0.25,
        share_delta_scale: float = 0.50,
    ) -> None:
        super().__init__()
        self.S = int(seq_len_hist)
        self.F = int(seq_len_fut)
        self.num_zones = int(num_zones)
        self.in_channels = int(in_channels)
        self.grid_size = tuple(grid_size)
        self.num_tiles = int(self.grid_size[0] * self.grid_size[1])
        self.weather_norm_clip = None if weather_norm_clip is None else float(weather_norm_clip)
        self.weather_encoder_force_fp32 = bool(weather_encoder_force_fp32)
        self.weather_history_steps = max(1, int(weather_history_steps))
        self.ramp_start_hour = int(ramp_start_hour)
        self.ramp_end_hour = int(ramp_end_hour)
        self.hour_band_size = max(1, int(hour_band_size))
        self.num_hour_bands = int(math.ceil(24 / float(self.hour_band_size)))
        self.log_residual_scale = float(log_residual_scale)
        self.system_log_residual_scale = float(system_log_residual_scale)
        self.share_delta_scale = float(share_delta_scale)

        self.model_config = {
            "in_channels": in_channels,
            "num_zones": num_zones,
            "seq_len_hist": seq_len_hist,
            "seq_len_fut": seq_len_fut,
            "weather_dim": weather_dim,
            "tab_dim": tab_dim,
            "decoder_dim": decoder_dim,
            "grid_size": tuple(grid_size),
            "num_future_layers": num_future_layers,
            "num_heads": num_heads,
            "head_hidden": head_hidden,
            "dropout": dropout,
            "weather_norm_clip": weather_norm_clip,
            "weather_encoder_force_fp32": weather_encoder_force_fp32,
            "later_horizon_start_step": later_horizon_start_step,
            "later_horizon_strength": later_horizon_strength,
            "later_horizon_power": later_horizon_power,
            "ramp_start_hour": ramp_start_hour,
            "ramp_end_hour": ramp_end_hour,
            "hour_band_size": hour_band_size,
            "use_calendar_second_harmonics": use_calendar_second_harmonics,
            "zone_adapter_rank": zone_adapter_rank,
            "weather_history_steps": weather_history_steps,
            "log_residual_scale": log_residual_scale,
            "system_log_residual_scale": system_log_residual_scale,
            "share_delta_scale": share_delta_scale,
        }

        # Weather paths. Use one shared CNN so the mapping for historical/future weather is consistent.
        self.weather_frame_encoder = WeatherSpatialEncoder(
            in_channels=in_channels,
            out_dim=weather_dim,
            grid_size=self.grid_size,
            hidden_channels=64,
            dropout=dropout,
            force_fp32=weather_encoder_force_fp32,
        )
        self.hist_weather_rnn = nn.GRU(weather_dim, weather_dim, batch_first=True)
        self.fut_weather_rnn = nn.GRU(weather_dim, weather_dim, batch_first=True)
        self.weather_summary_encoder = WeatherSummaryEncoder(in_channels, weather_dim, dropout=dropout)
        self.future_weather_fusion = nn.Sequential(
            nn.Linear(weather_dim * 2, weather_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(weather_dim),
        )

        # Calendar and demand paths.
        self.calendar_encoder = CalendarFeatureEncoder(
            out_dim=tab_dim,
            raw_dim=4,
            dropout=dropout,
            ramp_start_hour=ramp_start_hour,
            ramp_end_hour=ramp_end_hour,
            hour_band_size=hour_band_size,
            use_second_harmonics=use_calendar_second_harmonics,
        )
        self.hist_tab_proj = nn.Sequential(
            nn.Linear(num_zones + tab_dim, tab_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(tab_dim),
        )
        self.hist_tab_rnn = nn.GRU(tab_dim, tab_dim, batch_first=True)
        self.demand_history_encoder = DemandHistoryEncoder(num_zones, tab_dim, dropout=dropout)
        self.seasonal_lag_encoder = SeasonalLagEncoder(num_zones, self.F, decoder_dim, dropout=dropout)

        self.global_hist_proj = nn.Sequential(
            nn.Linear(tab_dim + tab_dim + weather_dim, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(decoder_dim),
        )

        self.future_context_proj = nn.Sequential(
            nn.Linear(tab_dim + weather_dim + decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(decoder_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.future_context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_future_layers)

        self.horizon_emb = nn.Embedding(self.F, decoder_dim)
        self.zone_emb = nn.Embedding(self.num_zones, decoder_dim)
        self.zone_tile_prior = nn.Embedding(self.num_zones, self.num_tiles)
        nn.init.zeros_(self.zone_tile_prior.weight)

        self.baseline_query_proj = nn.Linear(1, decoder_dim)
        self.weather_query_proj = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(decoder_dim),
        )
        self.weather_key_proj = nn.Linear(weather_dim, decoder_dim)
        self.weather_val_proj = nn.Linear(weather_dim, decoder_dim)
        self.weather_attn_dropout = nn.Dropout(dropout)

        self.zone_future_fusion = nn.Sequential(
            nn.Linear(decoder_dim * 4, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(decoder_dim),
        )

        head_in_dim = decoder_dim + decoder_dim + 1  # fused + zone_emb + baseline_norm
        self.zone_log_residual_head = nn.Sequential(
            nn.Linear(head_in_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.share_delta_head = nn.Sequential(
            nn.Linear(head_in_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.zone_bias_log_residual = nn.Embedding(self.num_zones, 1)
        self.zone_bias_share = nn.Embedding(self.num_zones, 1)

        self.system_head = nn.Sequential(
            nn.Linear(decoder_dim + tab_dim + weather_dim + 1, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.blend_head = nn.Sequential(
            nn.Linear(decoder_dim + tab_dim + weather_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        self._init_residual_heads_to_baseline()

        self.register_buffer("norm_mean", None)
        self.register_buffer("norm_std", None)
        self.register_buffer("weather_mean", None)
        self.register_buffer("weather_std", None)
        self.return_denormalized = False

    def _init_residual_heads_to_baseline(self) -> None:
        # Zero final layers => first forward exactly reproduces seasonal baseline.
        for head in [self.zone_log_residual_head, self.share_delta_head, self.system_head]:
            final = head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        nn.init.zeros_(self.zone_bias_log_residual.weight)
        nn.init.zeros_(self.zone_bias_share.weight)
        # Blend head can be arbitrary at init because both branches equal baseline when residuals are zero.
        final = self.blend_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    # ----------------------------
    # Stats / normalization API
    # ----------------------------

    def set_weather_stats(self, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> None:
        if mean is None or std is None:
            self.weather_mean = None
            self.weather_std = None
            return
        self.weather_mean = mean.detach().float().view(1, 1, -1, 1, 1)
        self.weather_std = std.detach().float().view(1, 1, -1, 1, 1).clamp_min(1e-6)

    def set_target_stats(
        self,
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
        return_denormalized: bool = False,
    ) -> None:
        self.norm_mean = None if mean is None else mean.detach().float().view(1, 1, -1)
        self.norm_std = None if std is None else std.detach().float().view(1, 1, -1).clamp_min(1e-6)
        self.return_denormalized = bool(return_denormalized)

    def _normalize_weather(self, weather_seq: torch.Tensor) -> torch.Tensor:
        if self.weather_mean is None or self.weather_std is None:
            out = weather_seq.float() if weather_seq.is_floating_point() else weather_seq
        else:
            out = (weather_seq.float() - self.weather_mean.to(weather_seq.device)) / self.weather_std.to(weather_seq.device)
        if self.weather_norm_clip is not None:
            out = out.clamp(min=-self.weather_norm_clip, max=self.weather_norm_clip)
        return out

    def _to_raw_target(self, y_norm: torch.Tensor) -> torch.Tensor:
        if self.norm_mean is None or self.norm_std is None:
            return y_norm
        return y_norm * self.norm_std.to(y_norm.device) + self.norm_mean.to(y_norm.device)

    def _to_norm_target(self, y_raw: torch.Tensor) -> torch.Tensor:
        if self.norm_mean is None or self.norm_std is None:
            return y_raw
        return (y_raw - self.norm_mean.to(y_raw.device)) / self.norm_std.to(y_raw.device)

    # ----------------------------
    # Input adaptation for evaluator-style calls
    # ----------------------------

    def adapt_inputs(
        self,
        history_weather: torch.Tensor,
        history_energy: torch.Tensor,
        future_weather: torch.Tensor,
        future_time: torch.Tensor,
    ):
        """
        Converts evaluator-style raw inputs into this model's training-time tensors.

        Expected evaluator shapes are the same as the previous model file:
        - history_weather: [B, S, H, W, C]
        - history_energy:  [B, S, Z]
        - future_weather:  [B, F, H, W, C]
        - future_time:     [B, F] integer hours or [B] first future hour
        """
        device = history_weather.device
        raw_hist_len = int(history_weather.size(1))
        use_hist = min(self.S, raw_hist_len)

        h_weather = history_weather[:, -use_hist:]
        h_energy = history_energy[:, -use_hist:]

        if self.norm_mean is not None and self.norm_std is not None:
            mean = self.norm_mean.to(device)
            std = self.norm_std.to(device)
            h_energy = (h_energy - mean) / std

        h_weather = h_weather.permute(0, 1, 4, 2, 3).contiguous()
        f_weather = future_weather.permute(0, 1, 4, 2, 3).contiguous()
        weather_seq = torch.cat([h_weather, f_weather], dim=1)

        future_time = future_time.long()
        if future_time.ndim == 1:
            first_future = future_time
        else:
            first_future = future_time[:, 0]
        hist_start = first_future - use_hist
        full_hours = torch.stack([hist_start + i for i in range(use_hist + self.F)], dim=1)
        hour_of_day = torch.remainder(full_hours, 24)
        day_index = torch.floor_divide(full_hours, 24)
        # Dataset timestamps are UTC. Offset +3 preserves the previous implementation's mapping.
        day_of_week = torch.remainder(day_index + 3, 7)

        hour_angle = (2.0 * math.pi / 24.0) * hour_of_day.float()
        dow_angle = (2.0 * math.pi / 7.0) * day_of_week.float()
        cal = torch.stack(
            [
                torch.sin(hour_angle),
                torch.cos(hour_angle),
                torch.sin(dow_angle),
                torch.cos(dow_angle),
            ],
            dim=-1,
        )
        hist_cal = cal[:, :use_hist, :]
        fut_cal = cal[:, use_hist:, :]
        return weather_seq, h_energy, hist_cal, fut_cal

    # ----------------------------
    # Internal model pieces
    # ----------------------------

    def _encode_weather_sequence(
        self,
        weather_seq: torch.Tensor,
        temporal_encoder: nn.GRU,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, steps, chans, height, width = weather_seq.shape
        flat = weather_seq.reshape(bsz * steps, chans, height, width)
        tile_tokens, global_feats = self.weather_frame_encoder(flat)
        tile_tokens = tile_tokens.reshape(bsz, steps, tile_tokens.size(1), tile_tokens.size(2))
        global_feats = global_feats.reshape(bsz, steps, global_feats.size(-1))
        seq_out, hidden = temporal_encoder(global_feats)
        return tile_tokens, seq_out, hidden[-1]

    def _compute_zone_local_weather(
        self,
        future_context: torch.Tensor,
        future_tile_tokens: torch.Tensor,
        baseline_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = future_context.size(0)
        zone_ids = torch.arange(self.num_zones, device=future_context.device)
        zone_embed = self.zone_emb(zone_ids).view(1, 1, self.num_zones, -1)

        step_context = future_context.unsqueeze(2).expand(-1, -1, self.num_zones, -1)
        baseline_context = self.baseline_query_proj(baseline_norm.unsqueeze(-1))
        query = self.weather_query_proj(step_context + zone_embed.expand(bsz, self.F, -1, -1) + baseline_context)

        keys = self.weather_key_proj(future_tile_tokens)
        values = self.weather_val_proj(future_tile_tokens)
        scale = 1.0 / math.sqrt(float(keys.size(-1)))
        attn_logits = torch.einsum("bfzd,bfpd->bfzp", query, keys) * scale
        attn_logits = attn_logits + self.zone_tile_prior(zone_ids).view(1, 1, self.num_zones, self.num_tiles)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.weather_attn_dropout(attn)
        local_weather = torch.einsum("bfzp,bfpd->bfzd", attn, values)
        return local_weather, zone_embed.expand(bsz, self.F, -1, -1)

    def _positive_raw_baseline(self, baseline_norm: torch.Tensor) -> torch.Tensor:
        baseline_raw = self._to_raw_target(baseline_norm)
        # Demand should be positive. This clamp only prevents invalid log/percentage behavior.
        return baseline_raw.clamp_min(1e-3)

    # ----------------------------
    # Forward
    # ----------------------------

    def forward(
        self,
        weather_seq: torch.Tensor,
        hist_demand: torch.Tensor,
        hist_cal: torch.Tensor,
        fut_cal: torch.Tensor,
    ) -> torch.Tensor:
        hist_steps = int(hist_demand.size(1))
        fut_steps = int(fut_cal.size(1))
        if fut_steps != self.F:
            raise ValueError(f"Expected future horizon {self.F}, got {fut_steps}")
        if weather_seq.size(1) != hist_steps + fut_steps:
            raise ValueError(
                f"weather_seq length must equal hist_len + fut_len; got {weather_seq.size(1)} vs {hist_steps + fut_steps}"
            )

        weather_seq = self._normalize_weather(weather_seq)
        full_hist_weather = weather_seq[:, :hist_steps]
        fut_weather = weather_seq[:, hist_steps:]
        weather_hist_steps = min(self.weather_history_steps, hist_steps)
        hist_weather = full_hist_weather[:, -weather_hist_steps:]

        # CNN weather path: only a bounded amount of historical weather is encoded.
        _, _, hist_weather_global = self._encode_weather_sequence(hist_weather, self.hist_weather_rnn)
        fut_tile_tokens, fut_weather_seq, _ = self._encode_weather_sequence(fut_weather, self.fut_weather_rnn)
        fut_weather_summary = self.weather_summary_encoder(hist_weather, fut_weather)
        fut_weather_feat = self.future_weather_fusion(torch.cat([fut_weather_seq, fut_weather_summary], dim=-1))

        # Calendar + demand history path.
        hist_cal_feat = self.calendar_encoder(hist_cal)
        fut_cal_feat = self.calendar_encoder(fut_cal)
        hist_tab = self.hist_tab_proj(torch.cat([hist_demand, hist_cal_feat], dim=-1))
        _, hist_tab_hidden = self.hist_tab_rnn(hist_tab)
        hist_tab_global = hist_tab_hidden[-1]
        demand_global = self.demand_history_encoder(hist_demand)

        global_hist = self.global_hist_proj(torch.cat([hist_tab_global, demand_global, hist_weather_global], dim=-1))
        global_hist_exp = global_hist.unsqueeze(1).expand(-1, self.F, -1)

        future_context = self.future_context_proj(torch.cat([fut_cal_feat, fut_weather_feat, global_hist_exp], dim=-1))
        horizon_ids = torch.arange(self.F, device=weather_seq.device).view(1, self.F)
        future_context = future_context + self.horizon_emb(horizon_ids)
        future_context = self.future_context_encoder(future_context)

        # Seasonal baseline and aligned per-zone lag features.
        baseline_norm, lag_emb = self.seasonal_lag_encoder(hist_demand)
        baseline_raw = self._positive_raw_baseline(baseline_norm)

        zone_local_weather, zone_embed = self._compute_zone_local_weather(future_context, fut_tile_tokens, baseline_norm)
        step_context = future_context.unsqueeze(2).expand(-1, -1, self.num_zones, -1)
        fused_context = self.zone_future_fusion(torch.cat([step_context, zone_embed, zone_local_weather, lag_emb], dim=-1))

        baseline_expand = baseline_norm.unsqueeze(-1)
        head_input = torch.cat([fused_context, zone_embed, baseline_expand], dim=-1)
        zone_ids = torch.arange(self.num_zones, device=weather_seq.device)

        # Direct per-zone multiplicative residual path.
        zone_log_resid = self.zone_log_residual_head(head_input).squeeze(-1)
        zone_log_resid = zone_log_resid + self.zone_bias_log_residual(zone_ids).view(1, 1, self.num_zones)
        zone_log_resid = self.log_residual_scale * torch.tanh(zone_log_resid)
        pred_zone_raw = baseline_raw * torch.exp(zone_log_resid)

        # System total + zone share path. Initialized to reproduce baseline exactly.
        baseline_system_raw = baseline_raw.sum(dim=-1, keepdim=True).clamp_min(1e-3)
        baseline_share = (baseline_raw / baseline_system_raw).clamp_min(1e-8)
        baseline_system_norm = baseline_norm.mean(dim=-1, keepdim=True)
        system_in = torch.cat([future_context, fut_cal_feat, fut_weather_feat, baseline_system_norm], dim=-1)
        system_log_resid = self.system_log_residual_scale * torch.tanh(self.system_head(system_in))
        pred_system_raw = baseline_system_raw * torch.exp(system_log_resid)

        share_delta = self.share_delta_head(head_input).squeeze(-1)
        share_delta = share_delta + self.zone_bias_share(zone_ids).view(1, 1, self.num_zones)
        share_delta = self.share_delta_scale * torch.tanh(share_delta)
        share_logits = torch.log(baseline_share) + share_delta
        pred_share = torch.softmax(share_logits, dim=-1)
        pred_share_raw = pred_system_raw * pred_share

        # Blend both paths. Both equal the baseline at init, so this is stable.
        blend_in = torch.cat([future_context, fut_cal_feat, fut_weather_feat], dim=-1)
        blend = torch.sigmoid(self.blend_head(blend_in))  # [B, F, 1]
        pred_raw = blend * pred_zone_raw + (1.0 - blend) * pred_share_raw
        pred_norm = self._to_norm_target(pred_raw)

        if self.return_denormalized and self.norm_mean is not None and self.norm_std is not None:
            return pred_raw
        return pred_norm


# Backwards-compatible alias. The train file only needs get_model, but this keeps old imports usable.
ResidualHybridForecastModel = SeasonalResidualHybridForecastModel


# ============================================================
# get_model / checkpoint loading
# ============================================================


def _default_model_config(metadata: Dict[str, object]) -> Dict[str, object]:
    return {
        "in_channels": int(metadata.get("in_channels", metadata.get("n_weather_vars", 7))),
        "num_zones": int(metadata.get("num_zones", metadata.get("n_zones", 8))),
        "seq_len_hist": int(metadata.get("seq_len_hist", metadata.get("history_len", 24))),
        "seq_len_fut": int(metadata.get("seq_len_fut", metadata.get("future_len", 24))),
        "weather_dim": int(metadata.get("weather_dim", 128)),
        "tab_dim": int(metadata.get("tab_dim", 128)),
        "decoder_dim": int(metadata.get("decoder_dim", 128)),
        "grid_size": _as_tuple2(metadata.get("grid_size", (4, 4))),
        "num_future_layers": int(metadata.get("num_future_layers", 2)),
        "num_heads": int(metadata.get("num_heads", 4)),
        "head_hidden": int(metadata.get("head_hidden", 128)),
        "dropout": float(metadata.get("dropout", 0.1)),
        "weather_norm_clip": None
        if metadata.get("weather_norm_clip", 10.0) is None
        else float(metadata.get("weather_norm_clip", 10.0)),
        "weather_encoder_force_fp32": bool(metadata.get("weather_encoder_force_fp32", True)),
        "later_horizon_start_step": int(metadata.get("later_horizon_start_step", 8)),
        "later_horizon_strength": float(metadata.get("later_horizon_strength", 0.0)),
        "later_horizon_power": float(metadata.get("later_horizon_power", 1.5)),
        "ramp_start_hour": int(metadata.get("ramp_start_hour", 15)),
        "ramp_end_hour": int(metadata.get("ramp_end_hour", 19)),
        "hour_band_size": int(metadata.get("hour_band_size", 4)),
        "use_calendar_second_harmonics": bool(metadata.get("use_calendar_second_harmonics", True)),
        "zone_adapter_rank": int(metadata.get("zone_adapter_rank", 0)),
        "weather_history_steps": int(metadata.get("weather_history_steps", 24)),
        "log_residual_scale": float(metadata.get("log_residual_scale", 0.35)),
        "system_log_residual_scale": float(metadata.get("system_log_residual_scale", 0.25)),
        "share_delta_scale": float(metadata.get("share_delta_scale", 0.50)),
    }


def _candidate_checkpoint_paths(base_dir: Path, explicit_path: Optional[object]) -> Tuple[Path, ...]:
    if explicit_path is not None:
        return (Path(explicit_path),)
    return (
        base_dir / "adjusted_hybrid.pth",
        base_dir / "seasonal_hybrid.pth",
        base_dir / "model_adjusted_v4.pth",
        base_dir / "checkpoints" / "adjusted_hybrid_best.pth",
        base_dir / "checkpoints" / "adjusted_hybrid_latest.pth",
        base_dir / "checkpoints" / "seasonal_hybrid_best.pth",
        base_dir / "checkpoints" / "seasonal_hybrid_latest.pth",
    )


def _extract_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return dict(ckpt["model_state_dict"])
        if "state_dict" in ckpt:
            return dict(ckpt["state_dict"])
    return dict(ckpt)


def _load_matching_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[list, list, list]:
    """
    Load only keys that exist and match shape. This prevents old-architecture
    checkpoints from crashing evaluator imports while still loading valid v4 ckpts.
    """
    own_state = model.state_dict()
    filtered = {}
    skipped_shape = []
    for key, value in state_dict.items():
        if key in {"norm_mean", "norm_std", "weather_mean", "weather_std"}:
            continue
        if key not in own_state:
            continue
        if tuple(own_state[key].shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        filtered[key] = value
    result = model.load_state_dict(filtered, strict=False)
    missing = [k for k in result.missing_keys if not k.endswith("num_batches_tracked")]
    unexpected = [k for k in result.unexpected_keys if not k.endswith("num_batches_tracked")]
    return missing, unexpected, skipped_shape


def get_model(metadata: Dict[str, object]):
    load_checkpoint = bool(metadata.get("load_checkpoint", True))
    base_dir = Path(__file__).resolve().parent
    checkpoint_path = None
    ckpt = None

    if load_checkpoint:
        for path in _candidate_checkpoint_paths(base_dir, metadata.get("checkpoint_path", None)):
            if path.exists():
                checkpoint_path = path
                ckpt = torch.load(path, map_location="cpu")
                break

    model_config = _default_model_config(metadata)
    if ckpt is not None and isinstance(ckpt, dict) and "model_config" in ckpt:
        saved_config = dict(ckpt["model_config"])
        # Keep current/evaluator metadata authoritative for dimensions when supplied.
        saved_config.update({k: v for k, v in metadata.items() if k in model_config})
        model_config.update({k: saved_config[k] for k in model_config.keys() if k in saved_config})

    model = SeasonalResidualHybridForecastModel(**model_config)

    if ckpt is not None:
        state_dict = _extract_state_dict(ckpt)
        missing, unexpected, skipped_shape = _load_matching_state_dict(model, state_dict)
        if missing or unexpected or skipped_shape:
            print(
                "[WARNING] Non-strict checkpoint load. "
                f"missing={missing[:20]} unexpected={unexpected[:20]} skipped_shape={skipped_shape[:20]}"
            )

        return_denormalized = bool(metadata.get("return_denormalized", True))
        if isinstance(ckpt, dict) and "norm_mean" in ckpt and "norm_std" in ckpt:
            model.set_target_stats(ckpt["norm_mean"], ckpt["norm_std"], return_denormalized=return_denormalized)
            print("[*] Loaded target normalization stats from checkpoint.")
        else:
            model.return_denormalized = return_denormalized

        if isinstance(ckpt, dict) and "weather_mean" in ckpt and "weather_std" in ckpt:
            model.set_weather_stats(ckpt["weather_mean"], ckpt["weather_std"])
            print("[*] Loaded weather normalization stats from checkpoint.")
        print(f"[*] Loaded checkpoint from {checkpoint_path}")
    else:
        model.return_denormalized = bool(metadata.get("return_denormalized", False))
        if load_checkpoint:
            print("[WARNING] No checkpoint found. Returning an untrained SeasonalResidualHybridForecastModel.")

    return model
