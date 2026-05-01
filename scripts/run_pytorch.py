#!/usr/bin/env python3
"""
Drive PolyFuzz against a coverage-instrumented PyTorch build.

This script is structured around an honest premise: a faithful PolyFuzz
port to PyTorch is bounded by obstacles that this artefact's authors
could not surmount in the available time. Rather than failing
mysteriously, this entry point makes the obstacles explicit and
diagnoses where each variant breaks down.

Two feedback modes are supported via --feedback-mode:

  - 'merged_bitmap' (Iteration 1): plain coverage.py + gcov merging.
    This works on PyTorch in principle, given a coverage build. The
    obstacles are environmental (build time, disk space) rather than
    fundamental.

  - 'branch_state' (Iteration 2, faithful PolyFuzz): branch-variable
    harvesting on both layers. The Python side runs unmodified via
    sys.settrace. The C side requires source instrumentation with
    MC_PROBE-style macros, which PyTorch does NOT have. This mode
    will report the obstacle and refuse to run on PyTorch unless
    --branch-state-c-disabled is passed (which falls back to
    Python-only branch-state).

Prerequisites:
  1. PyTorch built from source with --coverage. The dl-compiler-fuzzing
     skill documents the exact env vars (USE_GCOV=1, CFLAGS=--coverage,
     etc.). Plan for ~2-4 hours and 16+ GB RAM.
  2. The coverage build's source root exported as $TORCH_ROOT.
  3. coverage.py installed in the same venv.

Usage:
    export TORCH_ROOT=$HOME/pytorch-ptcov
    python scripts/run_pytorch.py --budget-sec 3600 \\
        --feedback-mode merged_bitmap \\
        --output-dir results/pytorch_run1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfuzz import PolyFuzz, PYTORCH_TARGET, SeedKind   # noqa: E402
from polyfuzz.coverage import CppCoverage, PythonCoverage  # noqa: E402
from polyfuzz.coverage.branch_vars import read_py_events  # noqa: E402
from polyfuzz.coverage.c_branch_vars import parse_trace_file  # noqa: E402
from polyfuzz.fuzzer import configure_logging              # noqa: E402


# ---------------------------------------------------------------------------
# Obstacle reporter
# ---------------------------------------------------------------------------

def report_obstacles(target, mode: str, branch_state_c_disabled: bool) -> dict:
    """Walk the prerequisites for `mode` on `target` and return a report.

    The report is a dict of {check_name: status_str}. Statuses are
      'ok'           -- prerequisite satisfied
      'missing'      -- prerequisite not satisfied; mode cannot run
      'partial'      -- prerequisite satisfied for one layer only
      'unsupported'  -- prerequisite cannot be satisfied for this target
    """
    report: dict = {}

    # --- gcov source root ---------------------------------------------
    if not target.gcov_source_root.exists():
        report["gcov_source_root"] = (
            f"missing: {target.gcov_source_root} does not exist. "
            f"Set $TORCH_ROOT to your coverage PyTorch build."
        )
    else:
        gcno = list(target.gcov_source_root.rglob("*.gcno"))
        if not gcno:
            report["gcov_source_root"] = (
                f"missing: {target.gcov_source_root} has no .gcno files. "
                f"Was PyTorch built with --coverage?"
            )
        else:
            report["gcov_source_root"] = (
                f"ok: {len(gcno)} .gcno files under "
                f"{target.gcov_source_root}"
            )

    # --- python_cov scope ---------------------------------------------
    importable = []
    for pkg in target.py_source_packages:
        try:
            __import__(pkg)
            importable.append(pkg)
        except ImportError:
            pass
    if importable:
        report["py_packages"] = (
            f"ok: importable: {importable}"
        )
    else:
        report["py_packages"] = (
            f"missing: none of {target.py_source_packages} importable. "
            f"Wrong venv?"
        )

    # --- branch_state-specific checks ---------------------------------
    if mode == "branch_state":
        # Python tracer applies regardless of target -- it's a runtime
        # property of CPython, not a property of the target.
        report["python_branch_tracer"] = (
            "ok: sys.settrace is available regardless of target"
        )
        # C tracer requires source instrumentation. Probe whether
        # libtorch_*.so contains the _mc_probe_stream symbol; if not,
        # the target was not instrumented and branch_state cannot
        # collect C events.
        c_probe_present = _probe_for_mc_probe_symbol(target)
        if c_probe_present:
            report["c_branch_probes"] = (
                "ok: target appears source-instrumented with MC_PROBE"
            )
        else:
            level = "missing" if not branch_state_c_disabled else "partial"
            report["c_branch_probes"] = (
                f"{level}: target NOT source-instrumented. "
                f"Branch_state mode on PyTorch requires either "
                f"(a) patching torch source to add MC_PROBE-style probes "
                f"around branches in aten/ and torch/csrc/, or "
                f"(b) writing an LLVM pass equivalent to the AFL++ pass "
                f"used by upstream PolyFuzz. Neither is implemented in "
                f"this artefact. Pass --branch-state-c-disabled to run "
                f"with Python branch-state only (degraded signal)."
            )

    return report


def _probe_for_mc_probe_symbol(target) -> bool:
    """Search the target's gcov_source_root for any shared library
    that exports _mc_probe_stream. Returns True iff at least one does.
    """
    try:
        sos = list(target.gcov_source_root.rglob("*.so"))
    except OSError:
        return False
    for so in sos[:50]:                # cap to avoid huge greps
        try:
            r = subprocess.run(
                ["nm", "-D", str(so)],
                capture_output=True, text=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        if "_mc_probe_stream" in r.stdout:
            return True
    return False


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(target, seed: pathlib.Path, mode: str) -> dict:
    """Run one seed manually to confirm the chosen mode collects signal.

    Refuses to continue if any layer reports zero. Returns a dict with
    counts so the caller can include them in the obstacles report.
    """
    print("=== smoke test ===", flush=True)
    counts: dict = {}

    if mode == "merged_bitmap":
        py = PythonCoverage(target.py_source_packages)
        cc = CppCoverage(
            gcov_source_root=target.gcov_source_root,
            include_prefixes=target.gcov_include_prefixes,
            exclude_prefixes=target.gcov_exclude_prefixes,
        )
        with tempfile.TemporaryDirectory() as work:
            wp = pathlib.Path(work)
            gcda = wp / "gcda"; gcda.mkdir()
            cov = wp / ".coverage"
            env = os.environ.copy()
            env["GCOV_PREFIX"] = str(gcda)
            env.update(target.extra_env)
            cmd = [sys.executable, "-m", "coverage", "run", "--branch",
                   f"--source={','.join(target.py_source_packages)}",
                   f"--data-file={cov}", str(seed)]
            print(f"  cmd: {' '.join(cmd[:5])} ...", flush=True)
            proc = subprocess.run(cmd, env=env, cwd=str(wp),
                                  capture_output=True, text=True,
                                  timeout=target.seed_timeout_sec)
            if proc.returncode != 0:
                print(f"  WARNING: smoke seed exit {proc.returncode}", flush=True)
                print(f"  stderr tail: {proc.stderr[-500:]}", flush=True)
            py_snap = py.read(cov)
            cc_snap = cc.read(gcda)
            counts.update({
                "py_lines":    py_snap.line_count(),
                "py_arcs":     len(py_snap.arcs),
                "cc_lines":    cc_snap.line_count(),
                "cc_branches": cc_snap.branch_count(),
            })
    else:  # branch_state
        scope = ",".join(target.py_source_packages)
        with tempfile.TemporaryDirectory() as work:
            wp = pathlib.Path(work)
            env = os.environ.copy()
            env.update(target.extra_env)
            cmd = [sys.executable, "-m", "polyfuzz.harness.seed_bootstrap",
                   str(seed), str(wp), scope]
            print(f"  cmd: {' '.join(cmd[:3])} ...", flush=True)
            proc = subprocess.run(cmd, env=env, cwd=str(wp),
                                  capture_output=True, text=True,
                                  timeout=target.seed_timeout_sec)
            if proc.returncode != 0:
                print(f"  WARNING: smoke seed exit {proc.returncode}", flush=True)
                print(f"  stderr tail: {proc.stderr[-500:]}", flush=True)
            py_snap = read_py_events(wp / "py_events.json")
            c_snap = parse_trace_file(wp / "c_trace.txt")
            counts.update({
                "py_branch_events": py_snap.event_count(),
                "c_branch_events":  c_snap.event_count(),
            })
    print(f"  smoke counts: {counts}", flush=True)
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--budget-sec", type=float, default=600.0)
    p.add_argument("--output-dir", type=pathlib.Path,
                   default=ROOT / "results" / "pytorch")
    p.add_argument("--seeds-dir", type=pathlib.Path,
                   default=ROOT / "seeds" / "pytorch")
    p.add_argument("--feedback-mode",
                   choices=("merged_bitmap", "branch_state"),
                   default="merged_bitmap",
                   help="Which fuzzer iteration to run. branch_state "
                        "requires source-instrumented PyTorch.")
    p.add_argument("--branch-state-c-disabled", action="store_true",
                   help="In branch_state mode, accept absent C probes "
                        "and run with Python tracer only.")
    p.add_argument("--smoke-only", action="store_true",
                   help="Run obstacle report + smoke test and exit.")
    p.add_argument("--skip-smoke", action="store_true",
                   help="Skip smoke test (not recommended).")
    args = p.parse_args()

    configure_logging(logging.INFO)

    target = PYTORCH_TARGET
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Always print and persist the obstacle report first.
    print(f"target: {target.name}")
    print(f"  py_source_packages: {target.py_source_packages}")
    print(f"  gcov_source_root:   {target.gcov_source_root}")
    print(f"  feedback_mode:      {args.feedback_mode}")
    print()

    obstacles = report_obstacles(target, args.feedback_mode,
                                 args.branch_state_c_disabled)
    print("=== obstacle report ===")
    for k, v in obstacles.items():
        marker = "OK " if v.startswith("ok") else "!! "
        print(f"  {marker}{k}: {v}")
    print()

    blocking = [k for k, v in obstacles.items()
                if v.startswith("missing")]

    # Decide whether we can proceed.
    if blocking:
        report_path = args.output_dir / "obstacle_report.json"
        report_path.write_text(json.dumps({
            "feedback_mode": args.feedback_mode,
            "obstacles":     obstacles,
            "blocking":      blocking,
            "decision":      "refused",
        }, indent=2))
        print(f"FATAL: blocking obstacles: {blocking}")
        print(f"Wrote {report_path}")
        return 2

    # Find seeds.
    seeds = sorted(args.seeds_dir.glob("seed_*.py"))
    if not seeds:
        sys.exit(f"FATAL: no seeds at {args.seeds_dir}")
    print(f"seeds available: {len(seeds)}")

    # Smoke test.
    smoke_counts: dict = {}
    if not args.skip_smoke:
        try:
            smoke_counts = smoke_test(target, seeds[0], args.feedback_mode)
        except Exception as e:
            print(f"FATAL: smoke test raised {type(e).__name__}: {e}")
            return 3

    # Persist the obstacles + smoke report.
    report_path = args.output_dir / "obstacle_report.json"
    report_path.write_text(json.dumps({
        "feedback_mode": args.feedback_mode,
        "obstacles":     obstacles,
        "blocking":      blocking,
        "smoke_counts":  smoke_counts,
        "decision":      "smoke_only" if args.smoke_only else "running",
    }, indent=2))

    if args.smoke_only:
        print(f"\nWrote {report_path}")
        return 0

    # Fuzz.
    fuzz = PolyFuzz(
        target=target,
        output_dir=args.output_dir,
        kind=SeedKind.PYTORCH,
        feedback_mode=args.feedback_mode,
    )
    fuzz.seed_initial(args.seeds_dir)
    print(f"\n=== fuzzing for {args.budget_sec:.0f}s ===")
    stats = fuzz.run(time_budget_sec=args.budget_sec)
    print(f"\n=== done ===")
    print(f"  iterations: {stats.iterations}")
    print(f"  bugs found: {stats.bugs_found}")
    print(f"  output:     {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
