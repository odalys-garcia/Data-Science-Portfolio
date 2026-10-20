import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _cyclical_to_index(cal_feats: torch.Tensor, sin_idx: int, cos_idx: int, period: int) -> torch.Tensor:
    """
    Recover a discrete cyclic index from sin/cos features.
    Expects cal_feats[..., sin_idx] = sin(2*pi*k/period), cal_feats[..., cos_idx] = cos(2*pi*k/period).
    """
    angle = torch.atan2(cal_feats[..., sin_idx], cal_feats[..., cos_idx])
    frac = torch.remainder(angle / (2.0 * math.pi), 1.0)
    idx = torch.round(frac * float(period)).long() % period
    return idx


class WeatherChannelGate(nn.Module):
    """
    Light per-frame channel reweighting.
    This is a cheap way to let the model suppress redundant weather channels
    without changing the external input shape.
    """

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
        # x: (B, C, H, W)
        stats = x.mean(dim=(2, 3))
        delta = 0.5 * torch.tanh(self.fc2(self.act(self.fc1(self.norm(stats)))))
        scale = (1.0 + delta).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class WeatherSpatialEncoder(nn.Module):
    """
    Compact per-frame spatial encoder.

    Unlike the previous version, this keeps tile tokens instead of collapsing each
    frame immediately into one global vector. That lets the decoder do zone-specific
    spatial weather selection later while still keeping the training-time footprint modest.
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
    def __init__(self, out_dim: int, raw_dim: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        emb_dim = max(8, out_dim // 8)
        self.hour_emb = nn.Embedding(24, emb_dim)
        self.dow_emb = nn.Embedding(7, emb_dim)
        self.weekend_emb = nn.Embedding(2, emb_dim)
        self.raw_proj = nn.Sequential(
            nn.Linear(raw_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_proj = nn.Sequential(
            nn.Linear((out_dim // 2) + 3 * emb_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, cal_feats: torch.Tensor) -> torch.Tensor:
        hour_idx = _cyclical_to_index(cal_feats, 0, 1, 24)
        dow_idx = _cyclical_to_index(cal_feats, 2, 3, 7)
        weekend_idx = (dow_idx >= 5).long()

        raw = self.raw_proj(cal_feats)
        hour = self.hour_emb(hour_idx)
        dow = self.dow_emb(dow_idx)
        weekend = self.weekend_emb(weekend_idx)
        return self.out_proj(torch.cat([raw, hour, dow, weekend], dim=-1))


class HistoryRegimeEncoder(nn.Module):
    """
    Explicit recent-level / spread / trend conditioning.

    The advanced diagnostics suggested non-trivial temporal drift and regime change.
    We cannot change the public model input shape, so this module extracts a regime
    vector directly from the historical demand window.
    """

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
        # hist_demand: (B, S, Z), expected normalized
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
    """
    Updated residual hybrid model.

    What stays the same externally:
    - same adapt_inputs(...) signature
    - same forward(weather_seq, hist_demand, hist_cal, fut_cal)
    - same output shape (B, F, Z)
    - same normalization hooks used by the current training script

    What changed internally:
    - residual-over-prev-day baseline stays
    - historical vs future weather branches stay separate
    - future weather is now kept as tile tokens so each zone can attend to its own
      spatially relevant weather region at each forecast step
    - a light channel gate handles moderate weather redundancy
    - a regime encoder conditions the forecast on recent level / spread / trend,
      which is a practical way to react to year/regime drift without changing the
      public input format
    """

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
    ) -> None:
        super().__init__()
        self.S = int(seq_len_hist)
        self.F = int(seq_len_fut)
        self.num_zones = int(num_zones)
        self.in_channels = int(in_channels)
        self.num_tiles = int(grid_size[0] * grid_size[1])
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

        self.calendar_encoder = CalendarFeatureEncoder(out_dim=tab_dim, raw_dim=4, dropout=dropout)
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
        # weather_seq: (B, T, C, H, W)
        bsz, steps, chans, height, width = weather_seq.shape
        flat = weather_seq.reshape(bsz * steps, chans, height, width)
        tile_tokens, global_feats = frame_encoder(flat)
        tile_tokens = tile_tokens.reshape(bsz, steps, tile_tokens.size(1), tile_tokens.size(2))
        global_feats = global_feats.reshape(bsz, steps, global_feats.size(-1))
        seq_out, hidden = temporal_encoder(global_feats)
        return tile_tokens, seq_out, hidden[-1]

    def _build_prev_day_baseline(self, hist_demand: torch.Tensor) -> torch.Tensor:
        if hist_demand.size(1) >= self.F:
            return hist_demand[:, -self.F:, :]
        pad = hist_demand[:, -1:, :].expand(-1, self.F - hist_demand.size(1), -1)
        return torch.cat([pad, hist_demand], dim=1)[:, -self.F:, :]

    def adapt_inputs(
        self,
        history_weather: torch.Tensor,
        history_energy: torch.Tensor,
        future_weather: torch.Tensor,
        future_time: torch.Tensor,
    ):
        """
        Bridge evaluate.py inputs into the model's train-time forward format.
        evaluate.py provides:
          history_weather: (B, 168, H, W, C)
          history_energy : (B, 168, Z) raw MWh
          future_weather : (B, 24, H, W, C)
          future_time    : (B, 24) hours since Unix epoch
        """
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
        # future_context: (B, F, D)
        # future_tile_tokens: (B, F, P, W)
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
        bsz = weather_seq.size(0)
        hist_steps = hist_demand.size(1)
        fut_steps = fut_cal.size(1)
        if fut_steps != self.F:
            raise ValueError(f"Expected future horizon {self.F}, got {fut_steps}")
        if weather_seq.size(1) != hist_steps + fut_steps:
            raise ValueError("weather_seq length must equal hist_len + fut_len")

        weather_seq = self._normalize_weather(weather_seq)
        hist_weather = weather_seq[:, :hist_steps]
        fut_weather = weather_seq[:, hist_steps:]

        _, hist_weather_seq, hist_weather_global = self._encode_weather_sequence(
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

        hist_tab = self.hist_tab_proj(torch.cat([hist_demand, hist_cal_feat], dim=-1))
        _, hist_tab_hidden = self.hist_tab_rnn(hist_tab)
        hist_tab_global = hist_tab_hidden[-1]
        regime_feat = self.regime_encoder(hist_demand)

        global_hist = self.global_hist_proj(torch.cat([hist_tab_global, hist_weather_global, regime_feat], dim=-1))
        global_hist_exp = global_hist.unsqueeze(1).expand(-1, self.F, -1)

        future_context = self.future_context_proj(torch.cat([fut_cal_feat, fut_weather_seq, global_hist_exp], dim=-1))
        horizon_ids = torch.arange(self.F, device=weather_seq.device).view(1, self.F)
        future_context = future_context + self.horizon_emb(horizon_ids)
        future_context = self.future_context_encoder(future_context)

        baseline = self._build_prev_day_baseline(hist_demand)
        zone_local_weather, zone_embed = self._compute_zone_local_weather(future_context, fut_tile_tokens, baseline)

        step_context = future_context.unsqueeze(2).expand(-1, -1, self.num_zones, -1)
        fused_context = self.zone_future_fusion(torch.cat([step_context, zone_embed, zone_local_weather], dim=-1))

        baseline_expand = baseline.unsqueeze(-1)
        head_input = torch.cat([fused_context, zone_embed, baseline_expand], dim=-1)
        residual = self.residual_head(head_input).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(head_input).squeeze(-1))
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
    }



def get_model(metadata: Dict[str, object]):
    """
    Compatible with evaluate.py and the current training script structure.

    Behavior:
    - evaluate.py calls get_model(metadata) and expects checkpoint loading to happen here.
    - train.py can pass load_checkpoint=False to create a fresh model while still using the same factory.
    - if a checkpoint exists, saved model_config inside the checkpoint overrides constructor defaults,
      which keeps evaluation compatible even when architecture hyperparameters change.
    """
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
        model.load_state_dict(state_dict, strict=True)

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
