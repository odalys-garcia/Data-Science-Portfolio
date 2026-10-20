from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# ============================================================
# Paths
# ============================================================
WEATHER_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/weather_data"
ENERGY_DIR = "/cluster/tufts/c26sp1cs0137/data/assignment3_data/energy_demand_data"

# Years available for training/validation. 2024 is held-out test data.
TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]

# ============================================================
# Selected input channels
# Leave empty to use all channels from the weather tensors.
# ============================================================
SELECTED_VARS: List[str] = []

MANIFEST_VERSION = 2


# ============================================================
# Helpers
# ============================================================
def load_energy_targets(energy_dir: Path, years: List[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        fpath = energy_dir / f"target_energy_zonal_{year}.csv"
        if not fpath.exists():
            print(f"[WARNING] Energy CSV not found, skipping: {fpath}")
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


def compute_normalization_stats(targets_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(targets_df.mean(axis=0).values, dtype=torch.float32)
    std = torch.tensor(targets_df.std(axis=0).values, dtype=torch.float32)
    std = std.clamp(min=1e-6)
    return mean, std


def _selected_vars_key(selected_vars: Sequence[str]) -> str:
    if not selected_vars:
        return "all"
    text = "|".join(map(str, selected_vars))
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _years_key(years: Sequence[int]) -> str:
    return "_".join(map(str, years))


def get_default_manifest_dir(weather_dir: Path) -> Path:
    return Path(weather_dir).resolve().parent / "manifests"


def _selected_idx_key(selected_idx: Optional[Sequence[int]]) -> str:
    if selected_idx is None:
        return "all"
    return "_".join(map(str, selected_idx))


def _weather_manifest_signature(manifest: Dict) -> str:
    payload = {
        "version": int(manifest.get("version", -1)),
        "weather_dir": str(manifest.get("weather_dir", "")),
        "years": list(map(int, manifest.get("years", []))),
        "selected_vars": list(map(str, manifest.get("selected_vars", []))),
        "selected_idx": None if manifest.get("selected_idx") is None else list(map(int, manifest.get("selected_idx", []))),
        "files": [],
    }
    files = manifest.get("files", {}) or {}
    for key in sorted(files.keys()):
        rec = files[key] or {}
        payload["files"].append(
            {
                "key": str(key),
                "path": str(rec.get("path", "")),
                "mtime_ns": int(rec.get("mtime_ns", -1)),
                "size": int(rec.get("size", -1)),
                "is_finite": bool(rec.get("is_finite", False)),
                "shape": None if rec.get("shape") is None else list(rec.get("shape")),
                "dtype": None if rec.get("dtype") is None else str(rec.get("dtype")),
                "error": None if rec.get("error") is None else str(rec.get("error")),
            }
        )
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _weather_manifest_path(manifest_dir: Path, years: Sequence[int], selected_vars: Sequence[str]) -> Path:
    return manifest_dir / f"weather_manifest__years-{_years_key(years)}__vars-{_selected_vars_key(selected_vars)}.pt"


def _window_manifest_path(
    manifest_dir: Path,
    years: Sequence[int],
    selected_vars: Sequence[str],
    seq_len_hist: int,
    seq_len_fut: int,
) -> Path:
    return manifest_dir / (
        f"window_manifest__years-{_years_key(years)}__vars-{_selected_vars_key(selected_vars)}"
        f"__S{seq_len_hist}__F{seq_len_fut}.pt"
    )


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


def resolve_selected_idx(weather_dir: Path, selected_vars: Sequence[str]) -> Optional[List[int]]:
    if not selected_vars:
        return None
    meta_path = Path(weather_dir).parent / "metadata.pt"
    if not meta_path.exists():
        print("[WARNING] metadata.pt not found. Using all channels.")
        return None
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    name_to_idx = {n: i for i, n in enumerate(meta["variable_names"])}
    return [name_to_idx[v] for v in selected_vars]


def safe_load_weather(fpath: Path) -> torch.Tensor:
    try:
        return torch.load(fpath, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(fpath, map_location="cpu", weights_only=True)


def build_weather_lookup(weather_dir: Path, years: Sequence[int]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for year in sorted(set(int(y) for y in years)):
        year_dir = Path(weather_dir) / str(year)
        if not year_dir.is_dir():
            continue
        for fpath in year_dir.glob("X_*.pt"):
            lookup[fpath.stem[2:]] = fpath
    return lookup


def build_or_load_weather_manifest(
    weather_dir: Path,
    years: Sequence[int],
    selected_vars: Sequence[str],
    selected_idx: Optional[List[int]] = None,
    manifest_dir: Optional[Path] = None,
    rebuild_manifest: bool = False,
    verbose: bool = True,
) -> Tuple[Dict, Path]:
    weather_dir = Path(weather_dir)
    manifest_dir = Path(manifest_dir or get_default_manifest_dir(weather_dir))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _weather_manifest_path(manifest_dir, years, selected_vars)

    current_lookup = build_weather_lookup(weather_dir, years)
    existing = None
    selected_idx_list = None if selected_idx is None else [int(v) for v in selected_idx]
    if manifest_path.exists() and not rebuild_manifest:
        try:
            existing = torch.load(manifest_path)
            if existing.get("version") != MANIFEST_VERSION:
                existing = None
            elif list(map(int, existing.get("years", []))) != list(map(int, years)):
                existing = None
            elif list(map(str, existing.get("selected_vars", []))) != list(map(str, selected_vars)):
                existing = None
            else:
                existing_selected_idx = existing.get("selected_idx")
                if existing_selected_idx is None:
                    existing_selected_idx_list = None
                else:
                    existing_selected_idx_list = [int(v) for v in existing_selected_idx]
                if existing_selected_idx_list != selected_idx_list:
                    existing = None
        except Exception:
            existing = None

    existing_files = existing.get("files", {}) if isinstance(existing, dict) else {}
    files_manifest: Dict[str, Dict] = {}
    reused = 0
    rescanned = 0
    bad = 0

    iterator = sorted(current_lookup.items())
    if verbose:
        iterator = tqdm(iterator, desc="Weather manifest")

    for key, fpath in iterator:
        stat = fpath.stat()
        old = existing_files.get(key)
        if (
            old is not None
            and old.get("path") == str(fpath)
            and int(old.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
            and int(old.get("size", -1)) == int(stat.st_size)
        ):
            files_manifest[key] = old
            reused += 1
            if not bool(old.get("is_finite", False)):
                bad += 1
            continue

        entry = {
            "path": str(fpath),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
            "is_finite": False,
            "shape": None,
            "dtype": None,
            "error": None,
        }
        try:
            x = safe_load_weather(fpath)
            if selected_idx is not None:
                x = x[..., selected_idx]
            if x.dtype != torch.float32:
                x = x.float()
            entry["shape"] = tuple(int(v) for v in x.shape)
            entry["dtype"] = str(x.dtype)
            entry["is_finite"] = bool(torch.isfinite(x).all().item())
        except Exception as exc:
            entry["error"] = repr(exc)
            entry["is_finite"] = False
        files_manifest[key] = entry
        rescanned += 1
        if not entry["is_finite"]:
            bad += 1

    manifest = {
        "version": MANIFEST_VERSION,
        "weather_dir": str(weather_dir),
        "years": list(map(int, years)),
        "selected_vars": list(selected_vars),
        "selected_idx": selected_idx_list,
        "num_files": len(files_manifest),
        "num_bad_files": bad,
        "reused_entries": reused,
        "rescanned_entries": rescanned,
        "files": files_manifest,
    }
    manifest["manifest_signature"] = _weather_manifest_signature(manifest)
    torch.save(manifest, manifest_path)
    if verbose:
        print(
            f"[manifest] Weather validity saved to {manifest_path} | files={len(files_manifest)} | "
            f"bad={bad} | reused={reused} | rescanned={rescanned}"
        )
    return manifest, manifest_path


def build_or_load_window_manifest(
    energy_df: pd.DataFrame,
    weather_lookup: Dict[str, Path],
    weather_manifest: Dict,
    years: Sequence[int],
    selected_vars: Sequence[str],
    seq_len_hist: int,
    seq_len_fut: int,
    target_nan: np.ndarray,
    manifest_dir: Path,
    rebuild_manifest: bool = False,
    verbose: bool = True,
) -> Tuple[Dict, Path]:
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _window_manifest_path(manifest_dir, years, selected_vars, seq_len_hist, seq_len_fut)

    weather_manifest_signature = str(
        weather_manifest.get("manifest_signature") or _weather_manifest_signature(weather_manifest)
    )

    if manifest_path.exists() and not rebuild_manifest:
        try:
            cached = torch.load(manifest_path)
            if (
                cached.get("version") == MANIFEST_VERSION
                and list(map(int, cached.get("years", []))) == list(map(int, years))
                and list(map(str, cached.get("selected_vars", []))) == list(map(str, selected_vars))
                and int(cached.get("seq_len_hist", -1)) == int(seq_len_hist)
                and int(cached.get("seq_len_fut", -1)) == int(seq_len_fut)
                and int(cached.get("num_rows", -1)) == int(len(energy_df))
                and str(cached.get("start", "")) == str(energy_df.index.min())
                and str(cached.get("end", "")) == str(energy_df.index.max())
                and str(cached.get("weather_manifest_signature", "")) == weather_manifest_signature
            ):
                if verbose:
                    print(f"[manifest] Loaded cached window manifest: {manifest_path}")
                return cached, manifest_path
        except Exception:
            pass

    keys = energy_df.index.strftime("%Y%m%d%H").tolist()
    weather_files = weather_manifest.get("files", {})
    step_valid = np.zeros(len(energy_df), dtype=bool)
    bad_weather_steps = 0
    missing_weather_steps = 0

    iterator = range(len(keys))
    if verbose:
        iterator = tqdm(iterator, desc="Window validity")

    for i in iterator:
        if target_nan[i]:
            continue
        key = keys[i]
        if key not in weather_lookup:
            missing_weather_steps += 1
            continue
        file_rec = weather_files.get(key)
        is_finite = bool(file_rec.get("is_finite", False)) if file_rec is not None else False
        if not is_finite:
            bad_weather_steps += 1
            continue
        step_valid[i] = True

    total_seq_len = int(seq_len_hist + seq_len_fut)
    timestamps = energy_df.index
    valid_start_indices: List[int] = []
    for i in range(len(energy_df) - total_seq_len + 1):
        j = i + total_seq_len - 1
        if not step_valid[i:i + total_seq_len].all():
            continue
        if (timestamps[j] - timestamps[i]) != timedelta(hours=total_seq_len - 1):
            continue
        valid_start_indices.append(int(i))

    manifest = {
        "version": MANIFEST_VERSION,
        "years": list(map(int, years)),
        "selected_vars": list(selected_vars),
        "seq_len_hist": int(seq_len_hist),
        "seq_len_fut": int(seq_len_fut),
        "num_rows": int(len(energy_df)),
        "start": str(energy_df.index.min()),
        "end": str(energy_df.index.max()),
        "weather_manifest_signature": weather_manifest_signature,
        "weather_manifest_num_files": int(weather_manifest.get("num_files", len(weather_manifest.get("files", {})))),
        "weather_manifest_num_bad_files": int(weather_manifest.get("num_bad_files", 0)),
        "is_valid_step": step_valid,
        "valid_start_indices": valid_start_indices,
        "num_bad_weather_steps": int(bad_weather_steps),
        "num_missing_weather_steps": int(missing_weather_steps),
        "num_valid_steps": int(step_valid.sum()),
        "num_valid_windows": int(len(valid_start_indices)),
    }
    torch.save(manifest, manifest_path)
    if verbose:
        print(
            f"[manifest] Window validity saved to {manifest_path} | valid_steps={int(step_valid.sum())} | "
            f"valid_windows={len(valid_start_indices)}"
        )
    return manifest, manifest_path


def prebuild_manifests(
    weather_dir: Path,
    energy_dir: Path,
    years_groups: Sequence[Sequence[int]],
    selected_vars: Sequence[str],
    seq_configs: Sequence[Tuple[int, int]],
    manifest_dir: Optional[Path] = None,
    rebuild_manifest: bool = False,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    weather_dir = Path(weather_dir)
    energy_dir = Path(energy_dir)
    manifest_dir = Path(manifest_dir or get_default_manifest_dir(weather_dir))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    selected_idx = resolve_selected_idx(weather_dir, selected_vars)

    created_paths: Dict[str, List[str]] = {"weather_manifests": [], "window_manifests": []}
    seen_weather_groups = set()

    for years in years_groups:
        years = tuple(sorted(set(map(int, years))))
        if not years:
            continue
        if years not in seen_weather_groups:
            weather_manifest, weather_path = build_or_load_weather_manifest(
                weather_dir=weather_dir,
                years=years,
                selected_vars=selected_vars,
                selected_idx=selected_idx,
                manifest_dir=manifest_dir,
                rebuild_manifest=rebuild_manifest,
                verbose=verbose,
            )
            created_paths["weather_manifests"].append(str(weather_path))
            seen_weather_groups.add(years)
        else:
            weather_manifest = torch.load(_weather_manifest_path(manifest_dir, years, selected_vars))

        energy_df = load_energy_targets(energy_dir, list(years))
        zone_cols = [c for c in energy_df.columns if c != "datetime"]
        target_nan = torch.isnan(torch.tensor(energy_df[zone_cols].values, dtype=torch.float32)).any(dim=1).numpy()
        weather_lookup = build_weather_lookup(weather_dir, years)

        for seq_len_hist, seq_len_fut in seq_configs:
            _, window_path = build_or_load_window_manifest(
                energy_df=energy_df,
                weather_lookup=weather_lookup,
                weather_manifest=weather_manifest,
                years=years,
                selected_vars=selected_vars,
                seq_len_hist=int(seq_len_hist),
                seq_len_fut=int(seq_len_fut),
                target_nan=target_nan,
                manifest_dir=manifest_dir,
                rebuild_manifest=rebuild_manifest,
                verbose=verbose,
            )
            created_paths["window_manifests"].append(str(window_path))

    return created_paths


# ============================================================
# Dataset
# ============================================================
class EnergyWeatherDataset(Dataset):
    """
    Multitask weather-to-energy dataset with a sliding window approach.
    Yields sequences of length S (history) + F (future).

    This manifest-enabled version persists:
    - per-weather-file finiteness checks
    - per-configuration valid window indices

    so future runs do not need to rescan every timestamp.
    """

    def __init__(
        self,
        weather_dir: Path,
        energy_dir: Path,
        years: List[int],
        selected_vars: List[str] = [],
        norm_mean: Optional[torch.Tensor] = None,
        norm_std: Optional[torch.Tensor] = None,
        scan_workers: int = 16,
        seq_len_hist: int = 24,
        seq_len_fut: int = 24,
        manifest_dir: Optional[Path] = None,
        use_manifest: bool = True,
        rebuild_manifest: bool = False,
        verbose: bool = True,
    ):
        self.weather_dir = Path(weather_dir)
        self.energy_dir = Path(energy_dir)
        self.years = list(map(int, years))
        self.S = int(seq_len_hist)
        self.F = int(seq_len_fut)
        self.total_seq_len = self.S + self.F
        self.verbose = bool(verbose)
        self.use_manifest = bool(use_manifest)
        self.rebuild_manifest = bool(rebuild_manifest)
        self.manifest_dir = Path(manifest_dir or get_default_manifest_dir(self.weather_dir))
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print("Loading energy demand CSVs …")
        self.energy_df = load_energy_targets(self.energy_dir, self.years)
        self.zone_names = [c for c in self.energy_df.columns if c != "datetime"]
        self.num_zones = len(self.zone_names)

        self.norm_mean = norm_mean
        self.norm_std = norm_std

        hours = self.energy_df.index.hour.values
        days = self.energy_df.index.dayofweek.values
        cal_features = np.column_stack([
            np.sin(2 * np.pi * hours / 24.0),
            np.cos(2 * np.pi * hours / 24.0),
            np.sin(2 * np.pi * days / 7.0),
            np.cos(2 * np.pi * days / 7.0),
        ])
        self.calendar_data = torch.tensor(cal_features, dtype=torch.float32)

        self.weather_file_lookup = build_weather_lookup(self.weather_dir, self.years)
        self.selected_idx = resolve_selected_idx(self.weather_dir, selected_vars)
        self.selected_vars = list(selected_vars)

        self._targets_raw = torch.tensor(self.energy_df[self.zone_names].values, dtype=torch.float32)
        self._target_nan = torch.isnan(self._targets_raw).any(dim=1).numpy()

        weather_manifest = None
        if self.use_manifest:
            weather_manifest, self.weather_manifest_path = build_or_load_weather_manifest(
                weather_dir=self.weather_dir,
                years=self.years,
                selected_vars=self.selected_vars,
                selected_idx=self.selected_idx,
                manifest_dir=self.manifest_dir,
                rebuild_manifest=self.rebuild_manifest,
                verbose=self.verbose,
            )
            window_manifest, self.window_manifest_path = build_or_load_window_manifest(
                energy_df=self.energy_df,
                weather_lookup=self.weather_file_lookup,
                weather_manifest=weather_manifest,
                years=self.years,
                selected_vars=self.selected_vars,
                seq_len_hist=self.S,
                seq_len_fut=self.F,
                target_nan=self._target_nan,
                manifest_dir=self.manifest_dir,
                rebuild_manifest=self.rebuild_manifest,
                verbose=self.verbose,
            )
            self.is_valid_step = np.asarray(window_manifest["is_valid_step"], dtype=bool)
            self.valid_start_indices = list(map(int, window_manifest["valid_start_indices"]))
            self._num_bad_weather_steps = int(window_manifest.get("num_bad_weather_steps", 0))
            self._num_missing_weather_steps = int(window_manifest.get("num_missing_weather_steps", 0))
            if self.verbose:
                print(
                    f"[manifest] Weather signature: {weather_manifest.get('manifest_signature', '<missing>')[:16]}... | "
                    f"window valid_steps={int(self.is_valid_step.sum())} | valid_windows={len(self.valid_start_indices)}"
                )
        else:
            self.weather_manifest_path = _weather_manifest_path(self.manifest_dir, self.years, self.selected_vars)
            self.window_manifest_path = _window_manifest_path(self.manifest_dir, self.years, self.selected_vars, self.S, self.F)
            self._weather_finite_cache: Dict[str, bool] = {}
            self._num_bad_weather_steps = 0
            self._num_missing_weather_steps = 0

            self.is_valid_step = np.zeros(len(self.energy_df), dtype=bool)
            keys = self.energy_df.index.strftime("%Y%m%d%H").tolist()
            if self.verbose:
                print("Validating individual timesteps...")
                iterator = tqdm(enumerate(keys), total=len(keys))
            else:
                iterator = enumerate(keys)
            for i, key in iterator:
                if self._target_nan[i]:
                    continue
                if key not in self.weather_file_lookup:
                    self._num_missing_weather_steps += 1
                    continue
                if self._weather_file_is_finite(self.weather_file_lookup[key]):
                    self.is_valid_step[i] = True
                else:
                    self._num_bad_weather_steps += 1

            if self.verbose:
                print("Constructing sliding windows...")
            self.valid_start_indices = []
            timestamps = self.energy_df.index
            for i in range(len(self.energy_df) - self.total_seq_len + 1):
                j = i + self.total_seq_len - 1
                if not self.is_valid_step[i:i + self.total_seq_len].all():
                    continue
                if (timestamps[j] - timestamps[i]) == timedelta(hours=self.total_seq_len - 1):
                    self.valid_start_indices.append(i)

        if self.verbose:
            if self._num_bad_weather_steps > 0:
                print(f"[WARNING] Skipped {self._num_bad_weather_steps} timesteps due to non-finite weather tensors.")
            if self._num_missing_weather_steps > 0:
                print(f"[WARNING] Skipped {self._num_missing_weather_steps} timesteps due to missing weather files.")
            print(f"Total valid sliding windows found: {len(self.valid_start_indices)}")

    def _select_channels(self, x: torch.Tensor) -> torch.Tensor:
        if self.selected_idx is None:
            return x
        return x[..., self.selected_idx]

    def _load_weather(self, fpath: Path) -> torch.Tensor:
        return safe_load_weather(fpath)

    def _weather_file_is_finite(self, fpath: Path) -> bool:
        cache_key = str(fpath)
        if hasattr(self, "_weather_finite_cache") and cache_key in self._weather_finite_cache:
            return self._weather_finite_cache[cache_key]

        try:
            x = self._load_weather(fpath)
            x = self._select_channels(x)
            if x.dtype != torch.float32:
                x = x.float()
            is_finite = bool(torch.isfinite(x).all().item())
        except Exception:
            is_finite = False

        if not hasattr(self, "_weather_finite_cache"):
            self._weather_finite_cache = {}
        self._weather_finite_cache[cache_key] = is_finite
        return is_finite

    def __len__(self) -> int:
        return len(self.valid_start_indices)

    def __getitem__(self, idx: int):
        start_idx = self.valid_start_indices[idx]
        end_idx = start_idx + self.total_seq_len

        seq_targets = self._targets_raw[start_idx:end_idx]
        seq_calendar = self.calendar_data[start_idx:end_idx]

        if self.norm_mean is not None and self.norm_std is not None:
            seq_targets = (seq_targets - self.norm_mean) / self.norm_std

        hist_demand = seq_targets[:self.S]
        future_demand = seq_targets[self.S:]
        hist_cal = seq_calendar[:self.S]
        fut_cal = seq_calendar[self.S:]

        keys = self.energy_df.index[start_idx:end_idx].strftime("%Y%m%d%H").tolist()
        weather_tensors = []
        for key in keys:
            fpath = self.weather_file_lookup[key]
            x = self._load_weather(fpath)
            x = self._select_channels(x)
            if x.dtype != torch.float32:
                x = x.float()
            if not torch.isfinite(x).all():
                raise ValueError(f"Encountered non-finite weather tensor during __getitem__: {fpath}")
            x = x.permute(2, 0, 1).contiguous()
            weather_tensors.append(x)

        weather_seq = torch.stack(weather_tensors, dim=0)
        return weather_seq, hist_demand, hist_cal, fut_cal, future_demand


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and/or inspect manifest-enabled dataset.")
    parser.add_argument("--weather-dir", type=str, default=WEATHER_DIR)
    parser.add_argument("--energy-dir", type=str, default=ENERGY_DIR)
    parser.add_argument("--years", type=str, default=",".join(map(str, TRAIN_YEARS)))
    parser.add_argument("--seq-len-hist", type=int, default=24)
    parser.add_argument("--seq-len-fut", type=int, default=24)
    parser.add_argument("--manifest-dir", type=str, default="")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--no-manifest", action="store_true")
    args = parser.parse_args()

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    ds = EnergyWeatherDataset(
        weather_dir=Path(args.weather_dir),
        energy_dir=Path(args.energy_dir),
        years=years,
        seq_len_hist=args.seq_len_hist,
        seq_len_fut=args.seq_len_fut,
        manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
        use_manifest=not args.no_manifest,
        rebuild_manifest=args.rebuild_manifest,
        verbose=True,
    )
    print(f"Dataset length: {len(ds)}")
