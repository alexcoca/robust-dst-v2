#!/usr/bin/env python3
"""average_metrics.py

Aggregate SGD‑X DST evaluation metrics **across seeds** and make it easy to
inspect results.

Improvements in this version
----------------------------
* Keeps all previous behaviour (per‑variant seed averaging, pretty table for
  selected services).
* **Adds a summary table** for the special aggregate categories
  `#ALL_SERVICES`, `#SEEN_SERVICES`, `#UNSEEN_SERVICES` – averaged across
  *variants*.
* Lets you choose the metric to display with `--metric` (defaults to
  `joint_goal_accuracy`).

Usage examples
--------------
```bash
# write v*_avg_metrics.json and print both tables (metric defaults to JGA)
python average_metrics.py /path/to/sgd_turn averaged \
       --services Travel_1,RideSharing_2,Movies_1,Hotels_2,Weather_1,Services_1

# just show overall / seen / unseen for slot_tagging_f1
python average_metrics.py /path/to/sgd_turn - --metric slot_tagging_f1
```
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from rich.console import Console
from rich.table import Table

JsonDict = Dict[str, Any]
console = Console()

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def find_variant_files(seed_dir: Path, version: int = 1) -> Dict[str, Path]:
    """Return mapping variant -> metrics json file for a single seed."""
    variant_files: Dict[str, Path] = {}
    for v_dir in seed_dir.glob("v[0-9]*"):
        metrics_files = list(v_dir.glob(f"test/version_{version}/*_metrics.json"))
        if metrics_files:
            variant_files[v_dir.name] = metrics_files[0]
    return variant_files


def mean_dict(dicts: List[JsonDict]) -> JsonDict:
    """Recursively average numeric leaves of a list of equal‑structured dicts."""
    if not dicts:
        raise ValueError("No dictionaries to average")

    out: JsonDict = {}
    for key in dicts[0]:
        vals = [d[key] for d in dicts]
        first_val = vals[0]
        if isinstance(first_val, dict):
            out[key] = mean_dict(vals)  # type: ignore[arg-type]
        else:
            out[key] = sum(vals) / len(vals)
    return out


# pattern helpers -------------------------------------------------------------
_suffix_re = re.compile(r"_([0-9]+)$")


def strip_variant(name: str) -> str:
    """Remove the *last* digit of a numeric suffix: `Weather_13` → `Weather_1`."""
    if name.startswith("#"):
        return name
    m = _suffix_re.search(name)
    if not m:
        return name
    digits = m.group(1)
    if len(digits) <= 1:
        return name
    return name[: m.start(1)] + digits[:-1]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Average SGD‑X DST metrics across seeds")
    p.add_argument("root", type=Path, help="Directory that contains the seed_* folders")
    p.add_argument(
        "output",
        type=str,
        help="Output directory for averaged JSONs or '-' to skip writing files",
    )
    p.add_argument(
        "--services",
        type=str,
        help="Comma‑separated list of services for the pretty table (e.g. Travel_1,Weather_1).",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="joint_goal_accuracy",
        help="Metric key to display in the tables (default: joint_goal_accuracy)",
    )
    p.add_argument(
        "--version",
        type=int,
        default=9,
        help="Version of the data for which predictions were produced"
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# core logic
# -----------------------------------------------------------------------------

def collect_averaged(root: Path, version: int = 1) -> Dict[str, JsonDict]:
    """Return per‑variant averaged metrics (across seeds)."""
    seed_dirs = sorted(root.glob("seed_*"))
    if not seed_dirs:
        sys.exit(f"No seed_* folders found under {root}")

    variant_metrics: Dict[str, List[JsonDict]] = defaultdict(list)
    for seed_dir in seed_dirs:
        for variant, m_path in find_variant_files(seed_dir, version=version).items():
            with m_path.open() as f:
                variant_metrics[variant].append(json.load(f))

    return {v: mean_dict(ms) for v, ms in variant_metrics.items()}


def write_variant_jsons(averaged: Dict[str, JsonDict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant, metrics in averaged.items():
        with (out_dir / f"{variant}_avg_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
    console.print(
        f"[green]Written averaged metrics for {len(averaged)} variants to {out_dir}[/]"
    )


# -----------------------------------------------------------------------------
# rich tables
# -----------------------------------------------------------------------------

def fmt(val: float | str) -> str:
    if isinstance(val, str):
        return val
    if math.isnan(val):
        return "–"
    return f"{val:.3f}"


def build_service_table(
    averaged: Dict[str, JsonDict], services: Sequence[str], metric: str
) -> Table:
    variants = sorted(averaged.keys())
    table = Table(show_lines=True, title=f"{metric} (services, averaged across seeds)")
    table.add_column("Service", style="bold cyan")
    for v in variants:
        table.add_column(v, justify="right")
    table.add_column("Across Variants", justify="right", style="bold")

    for svc_pattern in services:
        row: List[str] = [svc_pattern]
        vals: List[float] = []
        for v in variants:
            metrics = averaged[v]
            matched = [m for name, m in metrics.items() if strip_variant(name) == svc_pattern]
            if matched and metric in matched[0]:
                val = matched[0][metric]
                row.append(fmt(val))
                vals.append(val)
            else:
                row.append("–")
        overall = sum(vals) / len(vals) if vals else float("nan")
        row.append(fmt(overall))
        table.add_row(*row)
    return table


def build_meta_table(averaged: Dict[str, JsonDict], metric: str) -> Table:
    variants = sorted(averaged.keys())
    categories = ["#ALL_SERVICES", "#SEEN_SERVICES", "#UNSEEN_SERVICES"]
    table = Table(show_lines=True, title=f"{metric} (meta categories)")
    table.add_column("Category", style="bold magenta")
    for v in variants:
        table.add_column(v, justify="right")
    table.add_column("Across Variants", justify="right", style="bold")

    for cat in categories:
        row: List[str] = [cat]
        vals: List[float] = []
        for v in variants:
            val = averaged[v][cat][metric]
            row.append(fmt(val))
            vals.append(val)
        overall = sum(vals) / len(vals)
        row.append(fmt(overall))
        table.add_row(*row)
    return table


# -----------------------------------------------------------------------------
# entry point
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    args = parse_args()
    averaged = collect_averaged(args.root.resolve(), version=args.version)

    # write JSON files unless user passed '-'
    if args.output != "-":
        write_variant_jsons(averaged, Path(args.output).resolve())

    # meta summary table (always)
    console.print(build_meta_table(averaged, args.metric))

    # services table (optional)
    if args.services:
        services = [s.strip() for s in args.services.split(",") if s.strip()]
        console.print()
        console.print(build_service_table(averaged, services, args.metric))
    elif args.output == "-":
        # If no files written and no services requested, dump whole dict
        json.dump(averaged, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
