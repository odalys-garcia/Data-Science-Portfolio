#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from generate_dataset import (
    WEATHER_DIR,
    ENERGY_DIR,
    TRAIN_YEARS,
    build_or_load_weather_manifest,
    prebuild_manifests,
)


def parse_years(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_seq_configs(text: str) -> List[Tuple[int, int]]:
    configs: List[Tuple[int, int]] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        s, f = piece.split(":")
        configs.append((int(s), int(f)))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weather validity and valid-window manifests ahead of training/smoke tests.")
    parser.add_argument("--weather-dir", type=str, default=WEATHER_DIR)
    parser.add_argument("--energy-dir", type=str, default=ENERGY_DIR)
    parser.add_argument("--train-years", type=str, default=",".join(map(str, TRAIN_YEARS[:-1])))
    parser.add_argument("--val-years", type=str, default=str(TRAIN_YEARS[-1]))
    parser.add_argument("--extra-year-groups", type=str, default="", help="Semicolon-separated extra year groups, e.g. '2019,2020,2021,2022,2023;2024'")
    parser.add_argument("--seq-configs", type=str, default="24:24,168:24")
    parser.add_argument("--manifest-dir", type=str, default="")
    parser.add_argument("--rebuild-manifest", action="store_true")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else None
    train_years = parse_years(args.train_years)
    val_years = parse_years(args.val_years)
    seq_configs = parse_seq_configs(args.seq_configs)

    year_groups: List[List[int]] = []
    if train_years:
        year_groups.append(train_years)
    if val_years:
        year_groups.append(val_years)
    if args.extra_year_groups.strip():
        for grp in args.extra_year_groups.split(";"):
            yrs = parse_years(grp)
            if yrs:
                year_groups.append(yrs)

    created = prebuild_manifests(
        weather_dir=Path(args.weather_dir),
        energy_dir=Path(args.energy_dir),
        years_groups=year_groups,
        selected_vars=[],
        seq_configs=seq_configs,
        manifest_dir=manifest_dir,
        rebuild_manifest=args.rebuild_manifest,
        verbose=True,
    )
    print("\nManifest build complete.")
    print(json.dumps(created, indent=2))


if __name__ == "__main__":
    main()
