#!/usr/bin/env python3
"""
Multi-trial comparison: PolyFuzz vs Python-only vs C/C++-only.

Runs each variant N times with different RNG seeds and aggregates:
  - bugs found (mean, max, distribution by class)
  - corpus size at end
  - C branches discovered
  - Python arcs discovered
  - first-bug-time

This is the evaluation that backs the central claim of the report:
unifying Python and C/C++ coverage signals finds bugs that neither
single-language fuzzer finds in the same budget. Single-trial numbers
are meaningless for fuzzers; we always quote distributions.

Usage:
    python scripts/multi_trial.py --trials 5 --budget-sec 60
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import shutil
import statistics
import sys
import time
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfuzz import PolyFuzz, MOCK_TARGET, SeedKind  # noqa: E402
from polyfuzz.fuzzer import FitnessWeights              # noqa: E402


VARIANTS = [
    # (name, feedback_mode, fitness_weights)
    #
    # The four variants form an ablation:
    #   1. cc_only         — C/C++ coverage only, no Python signal
    #   2. py_only         — Python coverage only, no C/C++ signal
    #   3. merged_bitmap   — Iteration 1: union of Python+C/C++ coverage in one
    #                        bitmap. The "naive merged coverage" baseline.
    #   4. branch_state    — Iteration 2: PolyFuzz-style branch-variable feedback.
    #                        Same Python and C signals, but harvested as
    #                        (branch, value-class) tuples instead of edge bits.
    ("cc_only", "merged_bitmap", FitnessWeights(
        w_py_line=0.0, w_py_arc=0.0,
        w_cc_line=0.5, w_cc_branch=1.0,
        w_bug_bonus=10.0,
    )),
    ("py_only", "merged_bitmap", FitnessWeights(
        w_py_line=0.5, w_py_arc=1.0,
        w_cc_line=0.0, w_cc_branch=0.0,
        w_bug_bonus=10.0,
    )),
    ("merged_bitmap", "merged_bitmap", FitnessWeights(
        w_py_line=0.5, w_py_arc=1.0,
        w_cc_line=0.5, w_cc_branch=1.0,
        w_bug_bonus=10.0,
    )),
    ("branch_state", "branch_state", FitnessWeights(
        w_py_event=1.0, w_c_event=1.0,
        w_bug_bonus=10.0,
    )),
]


def _first_bug_iter(log_path: pathlib.Path) -> int:
    """Return the iteration index of the first bug, or -1 if none."""
    if not log_path.exists():
        return -1
    with log_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("bug_class", "none") not in ("none", "hang"):
                return rec["iter"]
    return -1


def _bug_classes(bugs_dir: pathlib.Path) -> Dict[str, int]:
    """Count bugs by class — uses the directory name prefix."""
    out: Dict[str, int] = {}
    if not bugs_dir.exists():
        return out
    for d in bugs_dir.iterdir():
        if not d.is_dir():
            continue
        cls = d.name.split("_", 1)[0]
        # 'runtime' is split off "runtime_error_<id>" — restore it.
        if cls == "runtime":
            cls = "runtime_error"
        out[cls] = out.get(cls, 0) + 1
    return out


def _unique_bugs(bugs_dir: pathlib.Path) -> int:
    """Count *unique* bugs by stderr signature (first 80 chars)."""
    if not bugs_dir.exists():
        return 0
    sigs = set()
    for d in bugs_dir.iterdir():
        report = d / "report.json"
        if not report.exists():
            continue
        try:
            obj = json.loads(report.read_text())
        except json.JSONDecodeError:
            continue
        sig = (obj.get("bug_class", "") + "|"
               + obj.get("stderr_tail", "")[:80].strip())
        sigs.add(sig)
    return len(sigs)


def run_one_trial(
    variant_name: str,
    feedback_mode: str,
    weights: FitnessWeights,
    out_dir: pathlib.Path,
    seeds_dir: pathlib.Path,
    budget_sec: float,
    rng_seed: int,
) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    fuzz = PolyFuzz(
        target=MOCK_TARGET,
        output_dir=out_dir,
        kind=SeedKind.MOCK,
        feedback_mode=feedback_mode,
        fitness_weights=weights,
        seed=rng_seed,
    )
    fuzz.seed_initial(seeds_dir)
    t0 = time.time()
    fuzz.run(time_budget_sec=budget_sec)
    elapsed = time.time() - t0

    stats = fuzz.stats_dict()
    feedback = stats["feedback"]
    # Per-mode feedback fields differ. Flatten to a uniform schema by
    # using whichever fields are present, defaulting to 0.
    return {
        "variant":          variant_name,
        "feedback_mode":    feedback_mode,
        "rng_seed":         rng_seed,
        "elapsed_sec":      elapsed,
        "iterations":       stats["iterations"],
        "corpus_size":      stats["corpus"]["size"],
        "bugs_found":       stats["bugs_found"],
        "unique_bugs":      _unique_bugs(out_dir / "bugs"),
        "bug_classes":      _bug_classes(out_dir / "bugs"),
        "first_bug_iter":   _first_bug_iter(out_dir / "polyfuzz.log.jsonl"),
        "py_lines":         feedback.get("py_lines", 0),
        "py_arcs":          feedback.get("py_arcs", 0),
        "cc_lines":         feedback.get("cc_lines", 0),
        "cc_branches":      feedback.get("cc_branches", 0),
        "py_branch_events": feedback.get("py_branch_events", 0),
        "c_branch_events":  feedback.get("c_branch_events", 0),
    }


def aggregate(trials: List[dict]) -> dict:
    """Reduce a list of per-trial dicts to means + ranges."""
    def stat(field: str) -> dict:
        vals = [t[field] for t in trials
                if t.get(field, 0) >= 0 or field != "first_bug_iter"]
        if not vals:
            return {"mean": 0, "min": 0, "max": 0}
        return {
            "mean": round(statistics.mean(vals), 2),
            "min": min(vals),
            "max": max(vals),
            "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
        }

    found_bug = [t for t in trials if t["bugs_found"] > 0]
    return {
        "trials":              len(trials),
        "trials_finding_bug":  len(found_bug),
        "iterations":          stat("iterations"),
        "corpus_size":         stat("corpus_size"),
        "bugs_found":          stat("bugs_found"),
        "unique_bugs":         stat("unique_bugs"),
        "py_arcs":             stat("py_arcs"),
        "cc_branches":         stat("cc_branches"),
        "py_branch_events":    stat("py_branch_events"),
        "c_branch_events":     stat("c_branch_events"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials",     type=int,   default=5)
    p.add_argument("--budget-sec", type=float, default=60.0)
    p.add_argument("--output-dir", type=pathlib.Path,
                   default=ROOT / "results" / "multi_trial")
    p.add_argument("--seeds-dir",  type=pathlib.Path,
                   default=ROOT / "seeds" / "mock")
    p.add_argument("--base-seed",  type=int, default=0xBADC0FFEE,
                   help="trial i uses base_seed + i as RNG seed")
    p.add_argument("--variants",  nargs="+", default=None,
                   help="restrict to a subset of variants by name")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        VARIANTS if args.variants is None
        else [v for v in VARIANTS if v[0] in args.variants]
    )
    if not selected:
        print(f"no matching variants in {[v[0] for v in VARIANTS]}")
        return 1

    all_results: Dict[str, List[dict]] = {v[0]: [] for v in selected}

    for i in range(args.trials):
        seed = args.base_seed + i
        print(f"\n=== trial {i+1}/{args.trials} (rng_seed={seed:#x}) ===")
        for vname, fmode, weights in selected:
            print(f"  [{vname:<14}] running...", end=" ", flush=True)
            t0 = time.time()
            r = run_one_trial(
                variant_name=vname,
                feedback_mode=fmode,
                weights=weights,
                out_dir=args.output_dir / f"trial_{i:02d}_{vname}",
                seeds_dir=args.seeds_dir,
                budget_sec=args.budget_sec,
                rng_seed=seed,
            )
            cov_disp = (
                f"cc_br={r['cc_branches']:3d}" if fmode == "merged_bitmap"
                else f"c_ev={r['c_branch_events']:3d}"
            )
            print(f"{time.time()-t0:5.1f}s "
                  f"iters={r['iterations']:4d} "
                  f"bugs={r['bugs_found']:2d} "
                  f"uniq={r['unique_bugs']:2d} "
                  f"corpus={r['corpus_size']:3d} "
                  f"{cov_disp}")
            all_results[vname].append(r)

            # Checkpoint after every trial so partial runs survive.
            partial = args.output_dir / "partial.json"
            partial.write_text(json.dumps({
                "config": {
                    "trials":     args.trials,
                    "budget_sec": args.budget_sec,
                    "base_seed":  args.base_seed,
                    "completed":  i + 1,
                },
                "raw":        all_results,
            }, indent=2))

    # Aggregate.
    aggregated = {v: aggregate(trials) for v, trials in all_results.items()}

    print("\n" + "=" * 100)
    print(f"AGGREGATE over {args.trials} trials × {args.budget_sec:.0f}s budget")
    print("=" * 100)
    fmt = "{:<14} {:>8} {:>8} {:>10} {:>10} {:>14} {:>14}"
    print(fmt.format(
        "variant", "tr/found", "bugs", "uniq_bugs", "corpus",
        "cc_br | c_ev", "py_arcs|py_ev",
    ))
    print(fmt.format(
        "", "", "(mean)", "(mean)", "(mean)", "(mean)", "(mean)",
    ))
    print("-" * 100)
    for vname, fmode, _ in selected:
        agg = aggregated[vname]
        cc_or_c = (
            agg["cc_branches"]["mean"] if fmode == "merged_bitmap"
            else agg["c_branch_events"]["mean"]
        )
        py_or_py = (
            agg["py_arcs"]["mean"] if fmode == "merged_bitmap"
            else agg["py_branch_events"]["mean"]
        )
        print(fmt.format(
            vname,
            f"{agg['trials_finding_bug']}/{agg['trials']}",
            f"{agg['bugs_found']['mean']}",
            f"{agg['unique_bugs']['mean']}",
            f"{agg['corpus_size']['mean']}",
            f"{cc_or_c}",
            f"{py_or_py}",
        ))
    print("=" * 100)

    # Persist.
    out = args.output_dir / "aggregate.json"
    out.write_text(json.dumps({
        "config": {
            "trials":        args.trials,
            "budget_sec":    args.budget_sec,
            "base_seed":     args.base_seed,
        },
        "raw":         all_results,
        "aggregated":  aggregated,
    }, indent=2))
    print(f"\nFull results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
