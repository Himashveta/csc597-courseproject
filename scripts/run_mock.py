#!/usr/bin/env python3
"""
Run a single PolyFuzz session against the mock target.

This is the `make demo` entry point: it builds a sane PolyFuzz over
the mock compiler, runs it for a budget, and prints a summary. Useful
for verifying the artefact installs and runs end to end.

Usage:
    python scripts/run_mock.py --budget-sec 30
    python scripts/run_mock.py --budget-sec 30 --output-dir results/x
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfuzz import PolyFuzz, MOCK_TARGET, SeedKind  # noqa: E402
from polyfuzz.fuzzer import configure_logging          # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--budget-sec", type=float, default=30.0)
    p.add_argument("--output-dir", type=pathlib.Path,
                   default=ROOT / "results" / "demo")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--seeds-dir", type=pathlib.Path,
                   default=ROOT / "seeds" / "mock")
    p.add_argument("--feedback-mode",
                   choices=("merged_bitmap", "branch_state"),
                   default="branch_state",
                   help="Iteration 1 (merged_bitmap) or Iteration 2 (branch_state).")
    p.add_argument("--clean", action="store_true",
                   help="wipe output_dir before running")
    args = p.parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    configure_logging(logging.INFO)

    fuzz = PolyFuzz(
        target=MOCK_TARGET,
        output_dir=args.output_dir,
        kind=SeedKind.MOCK,
        feedback_mode=args.feedback_mode,
        seed=args.seed,
    )
    n = fuzz.seed_initial(args.seeds_dir)
    print(f"Initial corpus: {n} seeds (mode={args.feedback_mode})")

    print(f"Fuzzing for {args.budget_sec:.0f}s...")
    fuzz.run(time_budget_sec=args.budget_sec)
    summary = fuzz.stats_dict()

    print("\n--- summary ---")
    print(json.dumps(summary, indent=2))

    bug_count = summary["bugs_found"]
    if bug_count > 0:
        print(f"\nFound {bug_count} bugs. See {args.output_dir / 'bugs'}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
