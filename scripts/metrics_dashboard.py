#!/usr/bin/env python
"""
metrics_dashboard.py
====================

Rich‑powered CLI to:
1. **Aggregate** DST metrics across seeds for variants *original* & v1‑v5 (plus optional
   decoding‑method folders).
2. **Compare** *original* performance between **any two** (<experiment>, <method>) pairs and
   print *signed* deltas (no absolute value) — including a per‑service diff.
3. **Per‑service breakdowns** — printed for the baseline/`--method-a` **and** for every
   method supplied via `--methods`, giving service‑level performance on the *original*
   split (SGD) for each decoding strategy.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
SECTION_KEYS = {"ALL": "#ALL_SERVICES", "SEEN": "#SEEN_SERVICES", "UNSEEN": "#UNSEEN_SERVICES"}
SERVICE_RE = re.compile(r".+_\d+$")
console = Console()

BASE_COLS: Tuple[Tuple[str, str, str], ...] = (
    ("JGA (all)", "joint_goal_accuracy", "ALL"),
    ("JGA (seen)", "joint_goal_accuracy", "SEEN"),
    ("JGA (unseen)", "joint_goal_accuracy", "UNSEEN"),
)
EXTRA_COLS: Tuple[Tuple[str, str, str], ...] = (
    ("Cat JGA (all)", "joint_cat_accuracy", "ALL"),
    ("Cat JGA (seen)", "joint_cat_accuracy", "SEEN"),
    ("Cat JGA (unseen)", "joint_cat_accuracy", "UNSEEN"),
    ("NonCat JGA (all)", "joint_noncat_accuracy", "ALL"),
    ("NonCat JGA (seen)", "joint_noncat_accuracy", "SEEN"),
    ("NonCat JGA (unseen)", "joint_noncat_accuracy", "UNSEEN"),
)


class MetricAggregator:
    """Utility to average metrics across multiple JSON files."""

    def __init__(self, json_files: Sequence[Path]):
        self.files = list(json_files)
        if not self.files:
            raise ValueError("MetricAggregator initialized with zero files.")

    def _get_section(self, data: dict, section: str) -> Optional[dict]:
        return data.get(SECTION_KEYS[section])

    def mean_metric(self, metric: str, section: str = "ALL") -> Optional[float]:
        values: List[float] = []
        for jf in self.files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            sec = self._get_section(data, section)
            if sec and metric in sec:
                values.append(sec[metric])
        return mean(values) if values else None

    def per_service_mean(self, metric: str = "joint_goal_accuracy") -> Dict[str, float]:
        service_vals: Dict[str, List[float]] = {}
        for jf in self.files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if SERVICE_RE.match(k) and metric in v:
                    service_vals.setdefault(k, []).append(v[metric])
        return {k: mean(v) for k, v in service_vals.items() if v}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_metric_files(exp_dir: Path, method: str, variant: str) -> List[Path]:
    files: List[Path] = []
    for seed_dir in sorted(exp_dir.glob("seed_*")):
        parts = [seed_dir]
        if method:
            parts.append(method)
        parts.extend([variant, "test"])
        base = Path(*parts)
        found = list(base.glob("version_*/model_*_metrics.json")) or list(base.glob("model_*_metrics.json"))
        files.extend(found)
    return files


def format_val(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "–"


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_variant_table(exp_dir: Path, variants: Sequence[str], methods: Sequence[str], extra: bool):
    tbl = Table(title=f"Averaged Metrics — {exp_dir.name}", box=box.SIMPLE_HEAVY)
    tbl.add_column("method/variant", justify="left")
    cols = list(BASE_COLS) + (list(EXTRA_COLS) if extra else [])
    for label, *_ in cols:
        tbl.add_column(label, justify="right")
    for method in ([""] + list(methods)):
        for variant in variants:
            files = collect_metric_files(exp_dir, method, variant)
            if not files:
                continue
            agg = MetricAggregator(files)
            row = [f"{method or 'baseline'}/{variant}"]
            for _, m, sec in cols:
                row.append(format_val(agg.mean_metric(m, sec)))
            tbl.add_row(*row)
    console.print(tbl)


def build_service_table(exp_dir: Path, method: str):
    files = collect_metric_files(exp_dir, method, "original")
    if not files:
        console.print(f"[yellow]No 'original' metrics for {exp_dir.name}/{method or 'baseline'} — skipping.[/]")
        return
    agg = MetricAggregator(files)
    svc = agg.per_service_mean()
    tbl = Table(title=f"Per‑service JGA (original — {exp_dir.name}/{method or 'baseline'})", box=box.SIMPLE_HEAVY)
    tbl.add_column("service", justify="left")
    tbl.add_column("JGA", justify="right")
    for k, v in sorted(svc.items(), key=lambda kv: kv[1]):
        tbl.add_row(k, format_val(v))
    console.print(tbl)


def build_diff_tables(exp_a: Path, method_a: str, exp_b: Path, method_b: str, extra: bool):
    files_a = collect_metric_files(exp_a, method_a, "original")
    files_b = collect_metric_files(exp_b, method_b, "original")
    if not files_a or not files_b:
        console.print("[red]Both experiment/method combos must contain original metrics.[/]")
        return
    agg_a, agg_b = MetricAggregator(files_a), MetricAggregator(files_b)
    cols = list(BASE_COLS) + (list(EXTRA_COLS) if extra else [])
    tbl = Table(title=f"Original Δ (B−A): {exp_a.name}/{method_a or 'baseline'} → {exp_b.name}/{method_b or 'baseline'}", box=box.SIMPLE_HEAVY)
    tbl.add_column("metric", justify="left")
    tbl.add_column("A", justify="right")
    tbl.add_column("B", justify="right")
    tbl.add_column("Δ", justify="right")
    for label, m, sec in cols:
        a, b = agg_a.mean_metric(m, sec), agg_b.mean_metric(m, sec)
        if a is None or b is None:
            continue
        tbl.add_row(label, format_val(a), format_val(b), f"{(b - a):+.4f}")
    console.print(tbl)

    svc_a, svc_b = agg_a.per_service_mean(), agg_b.per_service_mean()
    if svc_a and svc_b:
        tbl_svc = Table(title="Per‑service Δ (B−A) — sorted", box=box.SIMPLE_HEAVY)
        tbl_svc.add_column("service", justify="left")
        tbl_svc.add_column("A", justify="right")
        tbl_svc.add_column("B", justify="right")
        tbl_svc.add_column("Δ", justify="right")
        for svc_name in sorted(set(svc_a) & set(svc_b), key=lambda s: svc_b[s]-svc_a[s]):
            tbl_svc.add_row(svc_name, format_val(svc_a[svc_name]), format_val(svc_b[svc_name]), f"{(svc_b[svc_name]-svc_a[svc_name]):+.4f}")
        console.print(tbl_svc)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate DST metrics and render Rich tables.")
    p.add_argument("path", nargs="?", default=".", help="Primary experiment directory (default: .)")
    p.add_argument("--methods", nargs="*", default=[], help="Additional decoding‑method folders for aggregation AND per‑service displays")
    p.add_argument("--extra-metrics", action="store_true", help="Include cat/noncat JGA columns (overall/seen/unseen)")
    p.add_argument("--method-a", default="", help="Method sub-folder for the primary experiment's baseline per‑service table (default: baseline)")
    p.add_argument("--compare", metavar="PATH", help="Second experiment directory for diff on original")
    p.add_argument("--method-b", default="", help="Method sub-folder for comparison experiment (default: baseline)")
    return p.parse_args()


def main():
    args = parse_args()
    exp_a = Path(args.path).expanduser().resolve()
    if not exp_a.is_dir():
        console.print(f"[red]Error:[/] {exp_a} is not a directory.")
        raise SystemExit(1)

    variants = ["original", "v1", "v2", "v3", "v4", "v5"]
    build_variant_table(exp_a, variants, args.methods, args.extra_metrics)

    # Baseline per‑service
    build_service_table(exp_a, args.method_a)
    # Per‑service for each extra method
    for meth in args.methods:
        if meth == args.method_a:
            continue  # already printed
        build_service_table(exp_a, meth)

    # Optional diff
    if args.compare:
        exp_b = Path(args.compare).expanduser().resolve()
        if not exp_b.is_dir():
            console.print(f"[red]Error:[/] {exp_b} is not a directory.")
            raise SystemExit(1)
        build_diff_tables(exp_a, args.method_a, exp_b, args.method_b, args.extra_metrics)


if __name__ == "__main__":
    main()
