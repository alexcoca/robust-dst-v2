#!/usr/bin/env python3
"""
slot_error_analysis.py  ·  slot-level error analytics for SGD / SGD-X
---------------------------------------------------------------------

The utility compares *hypotheses* (model predictions) against *references* and
reports, **per service–slot**, the counts of

* C   – at least one correct value predicted
* S   – substitution errors            (wrong value(s) predicted)
* FN  – false negatives                (value *missed*)
* FP  – false positives                (slot predicted when absent)

Slot-Error-Rate is
    SER = (S + FN + FP) / (C + S + FN + FP)

Folder layout expected
----------------------
refs_root/
    original|v1|…|v5/<split>/dialogues_*.json

hyps_root/
    seed_*/original|v1|…|v5/<split>/version_<N>/checkpoint_*/dialogues_*.json
            ↑                                 ↑
         any name                       use  --version N

CLI
---
$ ./slot_error_analysis.py                                                \
        --refs-root data/raw                                              \
        --hyps-root hyps/d3st_sgd_turn                                    \
        --variants v1 v2 v3                                               \
        --services Restaurants_2 Hotels_2                                 \
        --slots date time restaurant_name                                 \
        --split test                                                      \
        --version 9                                                       \
        --out slot_errors.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Literal

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

console = Console()

ALIASES: Dict[str, Dict[str, str]] = {}

def _load_schema(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


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

def build_aliases(
        raw_root: Path,
        variants: List[str]
) -> dict:
    """Return a nested dict  variant → service → {canon→var, '_rev': {var→canon}}."""
    orig_schema = _load_schema(raw_root / "original" / "test" / "schema.json")

    aliases = {}
    for v in variants:
        var_schema = _load_schema(raw_root / v / "test" / "schema.json")
        mapping: Dict[str, dict] = defaultdict(dict)
        reverse: Dict[str, dict] = defaultdict(dict)

        for svc_orig, svc_var in zip(orig_schema, var_schema):
            canon_srv = svc_orig["service_name"]
            for field in ("slots", "intents"):
                for item_o, item_v in zip(svc_orig[field], svc_var[field]):
                    canon_slot_or_intent = item_o["name"]    # canonical key
                    var_slot_or_intent = item_v["name"]
                    mapping[canon_srv][canon_slot_or_intent] = var_slot_or_intent
                    reverse[canon_srv][var_slot_or_intent] = canon_slot_or_intent

        aliases[v] = {**mapping, "_rev": reverse}
        # slots will be mapped to identity
        aliases["original"] = {"_rev": defaultdict(lambda: {})}

    return aliases

# ---------------------------------------------------------------------------#
#                               normalisation                                #
# ---------------------------------------------------------------------------#
def _norm(v: str) -> str:
    """Case-/whitespace-insensitive normalisation."""
    return v.strip().lower()


# ---------------------------------------------------------------------------#
#                          per-slot statistic logic                           #
# ---------------------------------------------------------------------------#
Stat = Counter  # alias for readability   {"tp", "sub", "fp", "fn"}


def _update(stat: Stat, ref: set[str], hyp: set[str]) -> None:
    """Update *stat* with a single slot comparison."""
    if not ref and not hyp:
        return                       # nothing to score

    if ref and not hyp:
        stat["fn"] += 1              # missed the slot entirely
        return

    if hyp and not ref:
        stat["fp"] += len(hyp)       # predicted a slot that’s not there
        return

    # both non-empty -------------------------------------------------------
    correct = hyp & ref
    wrong = hyp - ref

    if correct:
        stat["tp"] += 1              # count once if *any* correct value
    else:
        stat["sub"] += len(hyp)      # all values wrong

    # still penalise the extra wrong values
    stat["sub"] += len(wrong)


# ---------------------------------------------------------------------------#
#                            directory traversal                              #
# ---------------------------------------------------------------------------#
def _dialogue_shards(base: Path) -> Iterable[Path]:
    """Yield every dialogues_*.json inside *base* (recursively)."""
    yield from base.rglob("dialogues_*.json")


def _load_shard(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------#
#                              core comparison                                #
# ---------------------------------------------------------------------------#
def analyse_variant(
    ref_dir: Path,
    hyp_seed_dirs: List[Path],
    split: str,
    version: int,
    variant: Literal['v1', 'v2', 'v3', 'v4', 'v5', 'original'],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Return {service → slot → {tp,sub,fp,fn,ser}} aggregating *across seeds*.
    """

    # defaultdict(service) → defaultdict(slot) → Counter
    stats: Dict[str, Dict[str, Stat]] = defaultdict(lambda: defaultdict(Stat))

    # iterate *every* seed
    for seed_root in hyp_seed_dirs:
        hyp_dir = seed_root / variant / split / f"version_{version}"
        hyp_shards = sorted(_dialogue_shards(hyp_dir))
        ref_shards = sorted(_dialogue_shards(ref_dir / split))

        if not hyp_shards:
            console.print(f"[yellow]No shards in {hyp_dir}; skipping seed[/yellow]")
            continue
        assert len(hyp_shards) == len(ref_shards), "Shard count mismatch"

        # ----- shard-by-shard --------------------------------------------
        for h_path, r_path in zip(hyp_shards, ref_shards):
            hyp_dials = _load_shard(h_path)
            ref_dials = _load_shard(r_path)

            assert len(hyp_dials) == len(ref_dials), "Dialogue count mismatch"

            # ----- dialogue-by-dialogue ---------------------------------
            for hd, rd in zip(hyp_dials, ref_dials):
                assert hd["dialogue_id"] == rd["dialogue_id"]

                for h_turn, r_turn in zip(hd["turns"], rd["turns"]):
                    if r_turn["speaker"].upper() != "USER":
                        continue

                    for h_fr, r_fr in zip(h_turn["frames"], r_turn["frames"]):
                        canon_srv = strip_variant(r_fr["service"])
                        if canon_srv not in ALIASES[variant]["_rev"]:
                            console.print(
                                f"[red]Warning: no alias mapping for service {canon_srv} in variant {variant}[/red]"
                            )

                        alias_rev = ALIASES.get(
                            variant, {}
                        ).get("_rev", {}).get(
                            canon_srv, {}
                        )
                        # possibly map SGD-X schema elements to SGD schema elements
                        gt_sv = {
                            alias_rev.get(name, name): {_norm(v) for v in vals}
                            for name, vals in r_fr["state"]["slot_values"].items()
                        }
                        pr_sv = {
                            alias_rev.get(name, name): {_norm(v) for v in vals}
                            for name, vals in h_fr["state"]["slot_values"].items()
                        }
                        for slot in set(gt_sv) | set(pr_sv):
                            stat = stats[canon_srv][slot]
                            ref_vals = {_norm(v) for v in gt_sv.get(slot, [])}
                            hyp_vals = {_norm(v) for v in pr_sv.get(slot, [])}
                            _update(stat, ref_vals, hyp_vals)

    # convert Counters to final numbers + SER
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for srv, slot_dict in stats.items():
        out[srv] = {}
        for slot, c in slot_dict.items():
            denom_all = sum(c.values()) or 1
            n_ref = c["tp"] + c["sub"] + c["fn"]
            n_pred = c["tp"] + c["sub"] + c["fp"]
            out[srv][slot] = {
                "C": c["tp"],
                "S": c["sub"],
                "FN": c["fn"],
                "FP": c["fp"],
                "FNR": c["fn"] / n_ref if n_ref else 0.0,
                "FPR": c["fp"] / n_pred if n_pred else 0.0,
                "SubR": c["sub"] / n_pred if n_pred else 0.0,
                "SER": (c["sub"] + c["fn"] + c["fp"]) / denom_all,
            }
    return out


# ---------------------------------------------------------------------------#
#                               CLI + tables                                 #
# ---------------------------------------------------------------------------#
def build_dataframe(
    nested: Dict[str, Dict[str, Dict[str, float]]],
    variant: str
) -> pd.DataFrame:
    """Flatten one variant’s dict → DataFrame."""
    rows = []
    for srv, slot_dict in nested.items():
        for slot, m in slot_dict.items():
            rows.append(
                dict(
                    variant=variant,
                    service=srv,
                    slot=slot,
                    **m
                )
            )
    return pd.DataFrame(rows)


def pretty_table(df: pd.DataFrame) -> Table:
    tbl = Table(title="Slot-level Error Summary", box=box.SIMPLE_HEAVY)
    tbl.add_column("Var", style="cyan", no_wrap=True)
    tbl.add_column("Service", style="magenta")
    tbl.add_column("Slot", style="green")
    for col in ("C", "S", "FN", "FP", "FNR", "FPR", "SubR", "SER"):
        tbl.add_column(col, justify="right")

    for _, r in df.sort_values(["variant", "service", "slot"]).iterrows():
        tbl.add_row(
            str(r.variant), r.service, r.slot,
            *(f"{r[col]:.3f}" if col in {"SER", "FNR", "FPR", "SubR"} else str(int(r[col]))
              for col in ("C", "S", "FN", "FP", "FNR", "FPR", "SubR", "SER"))
        )
    return tbl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Slot-level error analysis for SGD / SGD-X")
    p.add_argument("--refs-root",  type=Path, required=True,
                   help="Root of references (data/raw)")
    p.add_argument("--hyps-root",  type=Path, required=True,
                   help="Root of *one experiment* (contains seed_* dirs)")
    p.add_argument("--variants",   nargs="+",
                   default=["original", "v1", "v2", "v3", "v4", "v5"],
                   help="Schema variants to analyse")
    p.add_argument("--version",    type=int, default=9,
                   help="Number in version_<N> (default: 9)")
    p.add_argument("--split",      default="test",
                   help="Data split (default: test)")
    p.add_argument("--services",   nargs="*", help="Restrict to these services")
    p.add_argument("--slots",      nargs="*", help="Restrict to these slots")
    p.add_argument("--out",        type=Path,
                   help="Save full per-slot CSV here")
    return p.parse_args()


def main(argv: List[str] | None = None) -> None:

    args = parse_args() if argv is None else parse_args(argv)
    global ALIASES
    ALIASES = build_aliases(args.refs_root, args.variants)
    seed_dirs = sorted(args.hyps_root.glob("seed_*"))
    if not seed_dirs:
        console.print(f"[red]No seed_* dirs in {args.hyps_root}[/red]")
        sys.exit(1)

    frames = []
    for v in args.variants:
        ref_dir = args.refs_root / v
        if not ref_dir.exists():
            console.print(f"[yellow]Missing refs for {v}; skipping[/yellow]")
            continue

        res = analyse_variant(
            ref_dir,
            seed_dirs,
            split=args.split,
            version=args.version,
            variant=v
        )
        frames.append(build_dataframe(res, v))

    if not frames:
        console.print("[red]No data analysed – check paths/variants[/red]")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)

    # optional filtering -------------------------------------------------
    if args.services:
        df = df[df.service.isin(args.services)]
    if args.slots:
        df = df[df.slot.isin(args.slots)]

    # show table ---------------------------------------------------------
    console.print(pretty_table(df))

    # optional CSV -------------------------------------------------------
    if args.out:
        df.to_csv(args.out, index=False)
        console.print(f"[green]Saved CSV to {args.out}[/green]")


if __name__ == "__main__":
    main()
