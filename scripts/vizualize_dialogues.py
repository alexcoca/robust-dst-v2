#!/usr/bin/env python3
# visualize_dialogues.py

"""
Visualise one or more SGD/SGD-X dialogues side by side with
their slot‐level ground truth vs. predictions, using Rich.

Usage examples:

# 1) To browse through *all* dialogues in variant "v3" (press ENTER to move to the next):
python visualize_dialogues.py \
    --refs-root data/raw \
    --hyps-root hyps/d3st/d3st_sgd_turn/seed_2023060402_d3st_sgd_turn \
    --variant v3

# 2) To inspect a single dialogue (e.g. "1_00005") in a given variant:
python visualize_dialogues.py \
    --refs-root data/raw \
    --hyps-root hyps/d3st/d3st_sgd_turn/seed_2023060402_d3st_sgd_turn \
    --variant original \
    --dialogue 1_00005
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rich import box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise SGD/SGD-X dialogues with slot‐level GT vs. prediction"
    )
    p.add_argument(
        "--refs-root",
        type=Path,
        required=True,
        help="Root directory of references (e.g. data/raw)",
    )
    p.add_argument(
        "--hyps-root",
        type=Path,
        required=True,
        help="Root directory of hypotheses (e.g. hyps/.../seed_*/)",
    )
    p.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Schema variant to inspect (e.g. original, v1, v2, … v5)",
    )
    p.add_argument(
        "--dialogue",
        type=str,
        default=None,
        help="Optional single dialogue_id to display (e.g. '1_00005'). If omitted, pages through all.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        help="Which split to look in (default: test).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# JSON‐loading and indexing
# ─────────────────────────────────────────────────────────────────────────────

def load_all_shards(root: Path) -> Dict[str, dict]:
    """
    Load every dialogues_*.json shard under `root` (e.g. data/raw/v2/test)
    and return a mapping: dialogue_id → dialogue‐object.
    """
    out: Dict[str, dict] = {}
    for shard_path in sorted(root.glob("**/dialogues_*.json")):
        with shard_path.open("r", encoding="utf-8") as f:
            dialogues = json.load(f)
        for dlg in dialogues:
            did = dlg["dialogue_id"]
            if did in out:
                raise RuntimeError(f"Duplicate dialogue_id {did} in {shard_path}")
            out[did] = dlg
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Slot‐comparison logic
# ─────────────────────────────────────────────────────────────────────────────

def normalise(v: str) -> str:
    return v.strip().lower()


def compare_slot_sets(
    gt_values: List[str], hyp_values: List[str]
) -> (int, int, int, int):
    """
    Given ground‐truth list of (possibly multiple) values and hypothesis list,
    return a tuple (C, S, FN, FP), where:

      • C  = “true positives” = count of hypothesis values that match one of the GT set
      • S  = “substitution errors” = count of hypothesis values not in GT
      • FN = “false negatives” = count of GT values not predicted
      • FP = “false positives” = count of predicted values when GT is empty
    """
    gt_set = {normalise(x) for x in gt_values}
    hyp_set = {normalise(x) for x in hyp_values}

    # no ground truth and no hypothesis → nothing to score
    if not gt_set and not hyp_set:
        return 0, 0, 0, 0

    # GT exists but no prediction → all GT are missed (FN)
    if gt_set and not hyp_set:
        return 0, 0, len(gt_set), 0

    # Hyp exists but no ground truth → all are FP
    if hyp_set and not gt_set:
        return 0, 0, 0, len(hyp_set)

    # Both non‐empty:
    true_positive = len(hyp_set & gt_set)
    substitution = len(hyp_set - gt_set)
    false_negative = len(gt_set - hyp_set)
    # We do NOT count FP separately here when GT is non‐empty; any extra hyp that is not in GT is counted as a “substitution”.
    return true_positive, substitution, false_negative, 0


def slot_status(gt_values: List[str], hyp_values: List[str]) -> (str, str):
    """
    Determine a human‐readable 'status' string and a rich‐colour for this slot comparison:
      • If at least one hyp value ∈ GT, we call it “CORRECT”  (green).
      • If hyp exists but none ∈ GT, we call it “SUB”  (red).
      • If GT exists but no hyp  → “FN” (yellow).
      • If hyp exists but no GT  → “FP” (magenta).
      • If both lists are empty → “–” (grey).
    """
    gt_set = {normalise(x) for x in gt_values}
    hyp_set = {normalise(x) for x in hyp_values}

    if not gt_set and not hyp_set:
        return "–", "grey50"
    if gt_set and not hyp_set:
        return "FN", "yellow"
    if hyp_set and not gt_set:
        return "FP", "magenta"
    # both non‐empty
    if hyp_set & gt_set:
        return "CORRECT", "green"
    else:
        return "SUB", "red"


# ─────────────────────────────────────────────────────────────────────────────
# Rendering one dialogue in Rich
# ─────────────────────────────────────────────────────────────────────────────

def render_dialogue(
    dialogue: dict,
    hyp_dialogue: dict,
    service_slot_alias: Optional[dict] = None,
):
    """
    Render a single dialogue, printing both SYSTEM and USER turns.
    For USER turns, display slot‐comparison tables; for SYSTEM turns, display the utterance.
    """
    console.rule(f"[bold cyan] dialogue_id = {dialogue['dialogue_id']} [/]")

    for ref_turn, hyp_turn in zip(dialogue["turns"], hyp_dialogue["turns"]):
        speaker = ref_turn.get("speaker", "").upper()
        utterance = ref_turn.get("utterance", "")

        if speaker == "SYSTEM":
            # Display SYSTEM utterance in its own panel
            console.print(
                Panel(Text(utterance, style="white on dark_green"), title="SYSTEM", expand=False)
            )
            console.print()  # blank line after system turn

        elif speaker == "USER":
            # Display USER utterance
            console.print(
                Panel(Text(utterance, style="white on dark_blue"), title="USER", expand=False)
            )

            # For each frame in the reference turn, find matching frame in hyp_turn by service
            for frame_idx, ref_frame in enumerate(ref_turn.get("frames", [])):
                srv = ref_frame.get("service")
                # Find the hypothesis frame with the same service, if it exists
                hyp_frame = None
                for candidate in hyp_turn.get("frames", []):
                    if candidate.get("service") == srv:
                        hyp_frame = candidate
                        break
                if hyp_frame is None:
                    # If none, assume empty state for that service
                    hyp_frame = {"state": {"slot_values": {}}}

                # Extract slot_values dicts (may be empty)
                ref_slot_values: dict = ref_frame.get("state", {}).get("slot_values", {})
                hyp_slot_values: dict = hyp_frame.get("state", {}).get("slot_values", {})

                # Build a Rich Table for this service/frame
                table = Table(
                    title=f"Service = [bold]{srv}[/]   , frame #{frame_idx}",
                    box=box.SIMPLE,
                    show_header=True,
                    header_style="bold magenta",
                    expand=False,
                )
                table.add_column("Slot", style="cyan", no_wrap=True)
                table.add_column("GT Values", style="green")
                table.add_column("Hyp Values", style="white")
                table.add_column("Status", style="bold")

                # Union of slots present in either GT or Hyp
                all_slots = sorted(set(ref_slot_values.keys()) | set(hyp_slot_values.keys()))

                for slot_name in all_slots:
                    # Map through alias if provided
                    if service_slot_alias:
                        slot_name_canon = service_slot_alias.get(slot_name, slot_name)
                    else:
                        slot_name_canon = slot_name

                    gt_vals = ref_slot_values.get(slot_name, [])
                    hyp_vals = hyp_slot_values.get(slot_name, [])

                    status_str, colour = slot_status(gt_vals, hyp_vals)
                    gt_str = " | ".join(str(x) for x in gt_vals) if gt_vals else "–"
                    hyp_str = " | ".join(str(x) for x in hyp_vals) if hyp_vals else "–"

                    table.add_row(
                        slot_name_canon,
                        gt_str,
                        hyp_str,
                        f"[{colour}]{status_str}[/{colour}]",
                    )

                console.print(table)
            console.print()  # blank line between USER turns

        else:
            # In case there are other speaker labels, just print raw
            console.print(f"[grey50]Unknown speaker {speaker}: {utterance}[/grey50]")
            console.print()

    console.rule()


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI loop
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args()

    refs_dir = args.refs_root / args.variant / args.split
    hyps_dir = args.hyps_root / args.variant / args.split

    if not refs_dir.exists():
        console.print(f"[red]ERROR[/red]: references directory does not exist: {refs_dir}")
        sys.exit(1)
    if not hyps_dir.exists():
        console.print(f"[red]ERROR[/red]: hypothesis directory does not exist: {hyps_dir}")
        sys.exit(1)

    # Load all reference and hypothesis dialogues into dicts: id → dialogue
    ref_by_id = load_all_shards(refs_dir)
    hyp_by_id = load_all_shards(hyps_dir)

    # Determine which dialogue IDs to show
    all_ids = sorted(ref_by_id.keys())
    if args.dialogue:
        if args.dialogue not in ref_by_id:
            console.print(f"[red]ERROR[/red]: Dialogue ID {args.dialogue} not found under {refs_dir}")
            sys.exit(1)
        all_ids = [args.dialogue]

    for did in all_ids:
        ref_dlg = ref_by_id[did]
        hyp_dlg = hyp_by_id.get(did)
        if hyp_dlg is None:
            console.print(f"[yellow]WARNING[/yellow]: No hypothesis found for dialogue {did}, skipping.")
            continue

        # If you have an alias‐mapping dict (SGD→SGD-X or vice versa), pass it here.
        service_slot_alias = None

        render_dialogue(ref_dlg, hyp_dlg, service_slot_alias)
        if not args.dialogue:
            console.print("[bold yellow]Press ENTER to view the next dialogue…[/bold yellow]")
            _ = input()

    console.print("[green]Finished visualising dialogues.[/green]")


if __name__ == "__main__":
    main()
