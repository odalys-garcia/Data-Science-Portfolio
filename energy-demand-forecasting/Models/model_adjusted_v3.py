import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _cyclical_to_index(cal_feats: torch.Tensor, sin_idx: int, cos_idx: int, period: int) -> torch.Tensor:
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


class WeatherChannelGate(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        hidden = max(8, min(64, in_channels * 4))
        self.norm = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = x.mean(dim=(2, 3))
        delta = 0.5 * torch.tanh(self.fc2(self.act(self.fc1(self.norm(stats)))))
        scale = (1.0 + delta).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class WeatherSpatialEncoder(nn.Module):
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
        tokens = feat.flatten(2).transpose(1, 2)
        token_emb = self.token_proj(tokens)
        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.amax(dim=1)
        global_emb = self.global_proj(torch.cat([mean_pool, max_pool], dim=-1))
        return token_emb, global_emb

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        use_fp32 = self.force_fp32 and x.is_floating_point()
        if use_fp32:
            with torch.autocast(device_type=x.device.type, enabled=False):
                token_emb, global_emb = self._forward_impl(x.float())
            return token_emb.float(), global_emb.float()
        return self._forward_impl(x)


class CalendarFeatureEncoder(nn.Module):
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

        harmonic_dim = 4 if self.use_second_harmonics else 0
        self.raw_proj = nn.Sequential(
            nn.Linear(raw_dim + harmonic_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_proj = nn.Sequential(
            nn.Linear((out_dim // 2) + 5 * emb_dim, out_dim),
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

        raw_parts = [cal_feats]
        if self.use_second_harmonics:
            sin_h = cal_feats[..., 0]
            cos_h = cal_feats[..., 1]
            sin_d = cal_feats[..., 2]
            cos_d = cal_feats[..., 3]
            second = torch.stack(
                [
                    2.0 * sin_h * cos_h,
                    (cos_h * cos_h) - (sin_h * sin_h),
                    2.0 * sin_d * cos_d,
                    (cos_d * cos_d) - (sin_d * sin_d),
                ],
                dim=-1,
            )
            raw_parts.append(second)

        raw = self.raw_proj(torch.cat(raw_parts, dim=-1))
        hour = self.hour_emb(hour_idx)
        dow = self.dow_emb(dow_idx)
        weekend = self.weekend_emb(weekend_idx)
        ramp = self.ramp_emb(ramp_idx)
        hour_band = self.hour_band_emb(hour_band_idx)
        return self.out_proj(torch.cat([raw, hour, dow, weekend, ramp, hour_band], dim=-1))


class HistoryRegimeEncoder(nn.Module):
    def __init__(self, num_zones: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_zones = int(num_zones)
        in_dim = (6 * self.num_zones) + 3
        hidden = max(out_dim, 64)
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
        recent_w = min(6, steps)
        day_w = min(24, steps)

        last = hist_demand[:, -1, :]
        full_mean = hist_demand.mean(dim=1)
        full_std = hist_demand.std(dim=1, unbiased=False)
        day_mean = hist_demand[:, -day_w:, :].mean(dim=1)
        recent_mean = hist_demand[:, -recent_w:, :].mean(dim=1)
        trend = last - hist_demand[:, 0, :]

        system_last = last.mean(dim=-1, keepdim=True)
        system_day_mean = day_mean.mean(dim=-1, keepdim=True)
        system_trend = trend.mean(dim=-1, keepdim=True)

        feat = torch.cat(
            [last, day_mean, full_mean, full_std, recent_mean, trend, system_last, system_day_mean, system_trend],
            dim=-1,
        )
        return self.proj(feat)


class ResidualHybridForecastModel(nn.Module):
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
        later_horizon_start_step: int = 8,
        later_horizon_strength: float = 0.35,
        later_horizon_power: float = 1.5,
        ramp_start_hour: int = 15,
        ramp_end_hour: int = 19,
        hour_band_size: int = 4,
        use_calendar_second_harmonics: bool = True,
        zone_adapter_rank: int = 8,
    ) -> None:
        super().__init__()
        self.S = int(seq_len_hist)
        self.F = int(seq_len_fut)
        self.num_zones = int(num_zones)
        self.in_channels = int(in_channels)
        self.num_tiles = int(grid_size[0] * grid_size[1])
        self.later_horizon_start_step = int(later_horizon_start_step)
        self.later_horizon_strength = float(later_horizon_strength)
        self.later_horizon_power = float(later_horizon_power)
        self.ramp_start_hour = int(ramp_start_hour)
        self.ramp_end_hour = int(ramp_end_hour)
        self.hour_band_size = max(1, int(hour_band_size))
        self.use_calendar_second_harmonics = bool(use_calendar_second_harmonics)
        self.zone_adapter_rank = max(0, int(zone_adapter_rank))
        self.num_hour_bands = int(math.ceil(24 / float(self.hour_band_size)))

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
        }

        self.weather_norm_clip = None if weather_norm_clip is None else float(weather_norm_clip)
        self.weather_encoder_force_fp32 = bool(weather_encoder_force_fp32)

        self.hist_weather_frame_encoder = WeatherSpatialEncoder(
            in_channels=in_channels,
            out_dim=weather_dim,
            grid_size=grid_size,
            hidden_channels=64,
            dropout=dropout,
            force_fp32=weather_encoder_force_fp32,
        )
        self.fut_weather_frame_encoder = WeatherSpatialEncoder(
            in_channels=in_channels,
            out_dim=weather_dim,
            grid_size=grid_size,
            hidden_channels=64,
            dropout=dropout,
            force_fp32=weather_encoder_force_fp32,
        )
        self.hist_weather_rnn = nn.GRU(weather_dim, weather_dim, batch_first=True)
        self.fut_weather_rnn = nn.GRU(weather_dim, weather_dim, batch_first=True)

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
        self.regime_encoder = HistoryRegimeEncoder(num_zones=num_zones, out_dim=tab_dim, dropout=dropout)

        self.global_hist_proj = nn.Sequential(
            nn.Linear(tab_dim + weather_dim + tab_dim, decoder_dim),
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
        )
        self.future_context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_future_layers)

        self.horizon_emb = nn.Embedding(self.F, decoder_dim)
        self.zone_emb = nn.Embedding(self.num_zones, decoder_dim)
        self.zone_tile_prior = nn.Embedding(self.num_zones, self.num_tiles)
        nn.init.zeros_(self.zone_tile_prior.weight)

        self.future_ramp_emb = nn.Embedding(2, decoder_dim)
        self.future_hour_band_emb = nn.Embedding(self.num_hour_bands, decoder_dim)

        self.later_horizon_input_proj = nn.Sequential(
            nn.Linear(decoder_dim + tab_dim + weather_dim, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(decoder_dim),
        )
        self.later_horizon_rnn = nn.GRU(decoder_dim, decoder_dim, batch_first=True)
        self.later_horizon_out = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, decoder_dim),
        )
        nn.init.zeros_(self.later_horizon_out[-1].weight)
        nn.init.zeros_(self.later_horizon_out[-1].bias)
        self.later_horizon_norm = nn.LayerNorm(decoder_dim)

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
            nn.Linear(decoder_dim * 3, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(decoder_dim),
        )

        head_in_dim = decoder_dim + decoder_dim + 1
        self.residual_head = nn.Sequential(
            nn.Linear(head_in_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(head_in_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        self.zone_residual_bias = nn.Embedding(self.num_zones, 1)
        self.zone_gate_bias = nn.Embedding(self.num_zones, 1)
        nn.init.zeros_(self.zone_residual_bias.weight)
        nn.init.zeros_(self.zone_gate_bias.weight)

        if self.zone_adapter_rank > 0:
            self.zone_adapter_proj = nn.Linear(head_in_dim, self.zone_adapter_rank, bias=False)
            self.zone_residual_adapter = nn.Embedding(self.num_zones, self.zone_adapter_rank)
            self.zone_gate_adapter = nn.Embedding(self.num_zones, self.zone_adapter_rank)
            nn.init.zeros_(self.zone_residual_adapter.weight)
            nn.init.zeros_(self.zone_gate_adapter.weight)
        else:
            self.zone_adapter_proj = None
            self.zone_residual_adapter = None
            self.zone_gate_adapter = None

        self.register_buffer("norm_mean", None)
        self.register_buffer("norm_std", None)
        self.register_buffer("weather_mean", None)
        self.register_buffer("weather_std", None)
        self.return_denormalized = False

    def set_weather_stats(self, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> None:
        if mean is None or std is None:
            self.weather_mean = None
            self.weather_std = None
            return
        self.weather_mean = mean.detach().float().view(1, 1, -1, 1, 1)
        self.weather_std = std.detach().float().view(1, 1, -1, 1, 1).clamp_min(1e-6)

    def set_target_stats(self, mean: Optional[torch.Tensor], std: Optional[torch.Tensor], return_denormalized: bool = False) -> None:
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

    def _encode_weather_sequence(
        self,
        weather_seq: torch.Tensor,
        frame_encoder: WeatherSpatialEncoder,
        temporal_encoder: nn.GRU,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, steps, chans, height, width = weather_seq.shape
        flat = weather_seq.reshape(bsz * steps, chans, height, width)
        tile_tokens, global_feats = frame_encoder(flat)
        tile_tokens = tile_tokens.reshape(bsz, steps, tile_tokens.size(1), tile_tokens.size(2))
        global_feats = global_feats.reshape(bsz, steps, global_feats.size(-1))
        seq_out, hidden = temporal_encoder(global_feats)
        return tile_tokens, seq_out, hidden[-1]

    def _build_prev_day_baseline(self, hist_demand: torch.Tensor) -> torch.Tensor:
        if hist_demand.size(1) >= self.F:
            return hist_demand[:, -self.F :, :]
        pad = hist_demand[:, -1:, :].expand(-1, self.F - hist_demand.size(1), -1)
        return torch.cat([pad, hist_demand], dim=1)[:, -self.F :, :]

    def _later_horizon_mask(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.F <= 1 or self.later_horizon_strength <= 0.0:
            return torch.zeros(1, self.F, 1, device=device, dtype=dtype)
        idx = torch.arange(self.F, device=device, dtype=torch.float32)
        start = float(max(0, min(self.F - 1, self.later_horizon_start_step)))
        denom = max(float(self.F - 1) - start, 1.0)
        progress = ((idx - start) / denom).clamp(min=0.0, max=1.0)
        mask = self.later_horizon_strength * progress.pow(self.later_horizon_power)
        return mask.view(1, self.F, 1).to(dtype=dtype)

    def adapt_inputs(
        self,
        history_weather: torch.Tensor,
        history_energy: torch.Tensor,
        future_weather: torch.Tensor,
        future_time: torch.Tensor,
    ):
        device = history_weather.device
        raw_hist_len = history_weather.size(1)
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
        hist_start = future_time[:, 0] - use_hist
        full_hours = torch.stack([hist_start + i for i in range(use_hist + self.F)], dim=1)
        hour_of_day = torch.remainder(full_hours, 24)
        day_index = torch.floor_divide(full_hours, 24)
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

    def _compute_zone_local_weather(
        self,
        future_context: torch.Tensor,
        future_tile_tokens: torch.Tensor,
        baseline: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = future_context.size(0)
        zone_ids = torch.arange(self.num_zones, device=future_context.device)
        zone_embed = self.zone_emb(zone_ids).view(1, 1, self.num_zones, -1)

        step_context = future_context.unsqueeze(2).expand(-1, -1, self.num_zones, -1)
        baseline_context = self.baseline_query_proj(baseline.unsqueeze(-1))
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

    def forward(
        self,
        weather_seq: torch.Tensor,
        hist_demand: torch.Tensor,
        hist_cal: torch.Tensor,
        fut_cal: torch.Tensor,
    ) -> torch.Tensor:
        hist_steps = hist_demand.size(1)
        fut_steps = fut_cal.size(1)
        if fut_steps != self.F:
            raise ValueError(f"Expected future horizon {self.F}, got {fut_steps}")
        if weather_seq.size(1) != hist_steps + fut_steps:
            raise ValueError("weather_seq length must equal hist_len + fut_len")

        weather_seq = self._normalize_weather(weather_seq)
        hist_weather = weather_seq[:, :hist_steps]
        fut_weather = weather_seq[:, hist_steps:]

        _, _, hist_weather_global = self._encode_weather_sequence(
            hist_weather,
            self.hist_weather_frame_encoder,
            self.hist_weather_rnn,
        )
        fut_tile_tokens, fut_weather_seq, _ = self._encode_weather_sequence(
            fut_weather,
            self.fut_weather_frame_encoder,
            self.fut_weather_rnn,
        )

        hist_cal_feat = self.calendar_encoder(hist_cal)
        fut_cal_feat = self.calendar_encoder(fut_cal)
        fut_hour_idx = _cyclical_to_index(fut_cal, 0, 1, 24)
        fut_ramp_idx = _hours_in_range(fut_hour_idx, self.ramp_start_hour, self.ramp_end_hour).long()
        fut_hour_band_idx = torch.clamp(fut_hour_idx // self.hour_band_size, max=self.num_hour_bands - 1)

        hist_tab = self.hist_tab_proj(torch.cat([hist_demand, hist_cal_feat], dim=-1))
        _, hist_tab_hidden = self.hist_tab_rnn(hist_tab)
        hist_tab_global = hist_tab_hidden[-1]
        regime_feat = self.regime_encoder(hist_demand)

        global_hist = self.global_hist_proj(torch.cat([hist_tab_global, hist_weather_global, regime_feat], dim=-1))
        global_hist_exp = global_hist.unsqueeze(1).expand(-1, self.F, -1)

        future_context = self.future_context_proj(torch.cat([fut_cal_feat, fut_weather_seq, global_hist_exp], dim=-1))
        horizon_ids = torch.arange(self.F, device=weather_seq.device).view(1, self.F)
        future_context = future_context + self.horizon_emb(horizon_ids)
        future_context = future_context + self.future_ramp_emb(fut_ramp_idx) + self.future_hour_band_emb(fut_hour_band_idx)
        future_context = self.future_context_encoder(future_context)

        later_in = self.later_horizon_input_proj(torch.cat([future_context, fut_cal_feat, fut_weather_seq], dim=-1))
        later_seq, _ = self.later_horizon_rnn(later_in)
        future_context = self.later_horizon_norm(
            future_context + self._later_horizon_mask(weather_seq.device, future_context.dtype) * self.later_horizon_out(later_seq)
        )

        baseline = self._build_prev_day_baseline(hist_demand)
        zone_local_weather, zone_embed = self._compute_zone_local_weather(future_context, fut_tile_tokens, baseline)

        step_context = future_context.unsqueeze(2).expand(-1, -1, self.num_zones, -1)
        fused_context = self.zone_future_fusion(torch.cat([step_context, zone_embed, zone_local_weather], dim=-1))

        baseline_expand = baseline.unsqueeze(-1)
        head_input = torch.cat([fused_context, zone_embed, baseline_expand], dim=-1)
        residual = self.residual_head(head_input).squeeze(-1)
        gate_logits = self.gate_head(head_input).squeeze(-1)

        zone_ids = torch.arange(self.num_zones, device=weather_seq.device)
        residual = residual + self.zone_residual_bias(zone_ids).view(1, 1, self.num_zones)
        gate_logits = gate_logits + self.zone_gate_bias(zone_ids).view(1, 1, self.num_zones)

        if self.zone_adapter_rank > 0 and self.zone_adapter_proj is not None:
            zone_adapter_feat = self.zone_adapter_proj(head_input)
            residual = residual + torch.einsum("bfzr,zr->bfz", zone_adapter_feat, self.zone_residual_adapter(zone_ids))
            gate_logits = gate_logits + torch.einsum("bfzr,zr->bfz", zone_adapter_feat, self.zone_gate_adapter(zone_ids))

        gate = torch.sigmoid(gate_logits)
        pred_norm = baseline + gate * residual

        if self.return_denormalized and self.norm_mean is not None and self.norm_std is not None:
            mean = self.norm_mean.to(pred_norm.device)
            std = self.norm_std.to(pred_norm.device)
            return pred_norm * std + mean
        return pred_norm



def _default_model_config(metadata: Dict[str, object]) -> Dict[str, object]:
    return {
        "in_channels": int(metadata.get("in_channels", metadata.get("n_weather_vars", 7))),
        "num_zones": int(metadata.get("num_zones", metadata.get("n_zones", 8))),
        "seq_len_hist": int(metadata.get("seq_len_hist", metadata.get("history_len", 24))),
        "seq_len_fut": int(metadata.get("seq_len_fut", metadata.get("future_len", 24))),
        "weather_dim": int(metadata.get("weather_dim", 128)),
        "tab_dim": int(metadata.get("tab_dim", 128)),
        "decoder_dim": int(metadata.get("decoder_dim", 128)),
        "grid_size": tuple(metadata.get("grid_size", (4, 4))),
        "num_future_layers": int(metadata.get("num_future_layers", 2)),
        "num_heads": int(metadata.get("num_heads", 4)),
        "head_hidden": int(metadata.get("head_hidden", 128)),
        "dropout": float(metadata.get("dropout", 0.1)),
        "weather_norm_clip": None if metadata.get("weather_norm_clip", 10.0) is None else float(metadata.get("weather_norm_clip", 10.0)),
        "weather_encoder_force_fp32": bool(metadata.get("weather_encoder_force_fp32", True)),
        "later_horizon_start_step": int(metadata.get("later_horizon_start_step", 8)),
        "later_horizon_strength": float(metadata.get("later_horizon_strength", 0.35)),
        "later_horizon_power": float(metadata.get("later_horizon_power", 1.5)),
        "ramp_start_hour": int(metadata.get("ramp_start_hour", 15)),
        "ramp_end_hour": int(metadata.get("ramp_end_hour", 19)),
        "hour_band_size": int(metadata.get("hour_band_size", 4)),
        "use_calendar_second_harmonics": bool(metadata.get("use_calendar_second_harmonics", True)),
        "zone_adapter_rank": int(metadata.get("zone_adapter_rank", 8)),
    }



def get_model(metadata: Dict[str, object]):
    load_checkpoint = bool(metadata.get("load_checkpoint", True))
    checkpoint_path = metadata.get("checkpoint_path", None)
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent / "checkpoints" / "model_1_v2_ema_accum_best.pth"
    else:
        checkpoint_path = Path(checkpoint_path)

    ckpt = None
    if load_checkpoint and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    model_config = _default_model_config(metadata)
    if ckpt is not None and isinstance(ckpt, dict) and "model_config" in ckpt:
        saved_config = dict(ckpt["model_config"])
        model_config.update(saved_config)

    model = ResidualHybridForecastModel(**model_config)

    if ckpt is not None:
        state_dict = dict(ckpt.get("model_state_dict", ckpt))
        for key in ["norm_mean", "norm_std", "weather_mean", "weather_std"]:
            state_dict.pop(key, None)
        load_result = model.load_state_dict(state_dict, strict=False)
        missing_keys = [k for k in load_result.missing_keys if not k.endswith("num_batches_tracked")]
        unexpected_keys = [k for k in load_result.unexpected_keys if not k.endswith("num_batches_tracked")]
        if missing_keys or unexpected_keys:
            print(f"[WARNING] Non-strict checkpoint load. missing={missing_keys} unexpected={unexpected_keys}")

        return_denormalized = bool(metadata.get("return_denormalized", True))
        if "norm_mean" in ckpt and "norm_std" in ckpt:
            model.set_target_stats(ckpt["norm_mean"], ckpt["norm_std"], return_denormalized=return_denormalized)
            print("[*] Loaded target normalization stats from checkpoint.")
        else:
            model.return_denormalized = return_denormalized

        if "weather_mean" in ckpt and "weather_std" in ckpt:
            model.set_weather_stats(ckpt["weather_mean"], ckpt["weather_std"])
            print("[*] Loaded weather normalization stats from checkpoint.")
        print(f"[*] Successfully loaded weights from {checkpoint_path}")
    else:
        model.return_denormalized = bool(metadata.get("return_denormalized", False))
        if load_checkpoint:
            print(f"[WARNING] Could not find {checkpoint_path}. Returning an untrained model.")

    return model
