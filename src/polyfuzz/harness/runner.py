"""Subprocess-based seed runner.

For each seed the runner does:
  1. Allocate a unique work directory under <output_dir>/runs/<seed_id>/.
  2. Set GCOV_PREFIX so gcov flushes there at exit.
  3. Set up the per-seed coverage data file.
  4. Choose between two execution modes (controlled by feedback_mode):

     - 'merged_bitmap' (Iteration 1): plain `coverage run --branch` over
        the seed file. Reads coverage.py + gcov afterward. The unified
        bitmap fitness signal.
     - 'branch_state' (Iteration 2): runs the seed under
        polyfuzz.harness.seed_bootstrap, which installs the Python
        branch-variable tracer and sets up MC_TRACE_FD for the C
        probes. Reads py_events.json + c_trace.txt afterward. The
        PolyFuzz-style branch-state fitness signal.
     - 'both' (default): runs the seed twice — once for coverage, once
        for branch-state. Used for the head-to-head evaluation.

  5. Run with a timeout, capture stdout/stderr.
  6. Return SeedRunOutcome with paths to all per-seed coverage data.

Why subprocess instead of in-process: gcov's flush is tied to process
exit (or __gcov_dump). Forking per-seed gives a clean flush boundary,
isolates SUT crashes from the fuzzer, and is the standard AFL pattern.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Optional

from polyfuzz.target import TargetSpec


@dataclasses.dataclass
class SeedRunOutcome:
    """Everything the fuzzer needs to know about one seed run."""
    seed_id: str
    return_code: int
    timed_out: bool
    duration_sec: float
    stdout_tail: str
    stderr_tail: str
    coverage_data_file: pathlib.Path     # for python_cov.py
    gcov_prefix_dir: pathlib.Path         # for cpp_cov.py
    py_events_file: pathlib.Path          # for branch_vars.py
    c_trace_file: pathlib.Path            # for c_branch_vars.py
    seed_file: pathlib.Path


class SeedRunner:
    """Runs one seed in a subprocess.

    feedback_mode controls which signals are collected:
      'merged_bitmap' -> coverage.py + gcov (Iteration 1)
      'branch_state'  -> py_events + c_trace (Iteration 2)
      'both'          -> all four; runs the seed twice
    """

    def __init__(
        self,
        target: TargetSpec,
        runs_dir: pathlib.Path,
        python_exe: Optional[str] = None,
        feedback_mode: str = "both",
    ) -> None:
        if feedback_mode not in ("merged_bitmap", "branch_state", "both"):
            raise ValueError(f"unknown feedback_mode: {feedback_mode}")
        self._target = target
        self._runs_dir = runs_dir.resolve()
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._python = python_exe or sys.executable
        self._feedback_mode = feedback_mode

    def run(
        self,
        seed_id: str,
        seed_path: pathlib.Path,
    ) -> SeedRunOutcome:
        seed_path = seed_path.resolve()
        work_dir = self._runs_dir / seed_id
        work_dir.mkdir(parents=True, exist_ok=True)
        gcov_prefix = (work_dir / "gcda").resolve()
        gcov_prefix.mkdir(parents=True, exist_ok=True)
        cov_data = (work_dir / ".coverage").resolve()
        py_events = (work_dir / "py_events.json").resolve()
        c_trace = (work_dir / "c_trace.txt").resolve()

        return_code = 0
        stdout = ""
        stderr = ""
        timed_out = False
        t0 = time.time()

        # Pass 1: coverage.py + gcov, if requested.
        if self._feedback_mode in ("merged_bitmap", "both"):
            rc, so, se, to = self._run_coverage_pass(
                seed_path, work_dir, gcov_prefix, cov_data,
            )
            return_code, stdout, stderr, timed_out = rc, so, se, to

        # Pass 2: branch-state, if requested.
        # If the coverage pass already crashed, we still run the
        # branch-state pass to capture as much signal as possible.
        if self._feedback_mode in ("branch_state", "both") and not timed_out:
            rc2, so2, se2, to2 = self._run_branch_state_pass(
                seed_path, work_dir, gcov_prefix,
            )
            # When mode is 'both', the merged_bitmap pass's outputs are
            # what we report; the branch_state pass is just for events.
            # When mode is 'branch_state' alone, this is THE pass.
            if self._feedback_mode == "branch_state":
                return_code, stdout, stderr, timed_out = rc2, so2, se2, to2

        elapsed = time.time() - t0
        return SeedRunOutcome(
            seed_id=seed_id,
            return_code=return_code,
            timed_out=timed_out,
            duration_sec=elapsed,
            stdout_tail=_tail(stdout, 4000),
            stderr_tail=_tail(stderr, 4000),
            coverage_data_file=cov_data,
            gcov_prefix_dir=gcov_prefix,
            py_events_file=py_events,
            c_trace_file=c_trace,
            seed_file=seed_path,
        )

    # ----------------------------------------------------------------

    def _run_coverage_pass(
        self,
        seed_path: pathlib.Path,
        work_dir: pathlib.Path,
        gcov_prefix: pathlib.Path,
        cov_data: pathlib.Path,
    ) -> tuple:
        cmd = [
            self._python,
            "-m", "coverage", "run",
            "--branch",
            f"--source={','.join(self._target.py_source_packages)}",
            f"--data-file={cov_data}",
            str(seed_path),
        ]
        env = self._build_env(gcov_prefix, with_c_trace=False)
        return self._run_cmd(cmd, env, work_dir)

    def _run_branch_state_pass(
        self,
        seed_path: pathlib.Path,
        work_dir: pathlib.Path,
        gcov_prefix: pathlib.Path,
    ) -> tuple:
        scope_csv = ",".join(self._target.py_source_packages)
        cmd = [
            self._python,
            "-m", "polyfuzz.harness.seed_bootstrap",
            str(seed_path),
            str(work_dir),
            scope_csv,
        ]
        env = self._build_env(gcov_prefix, with_c_trace=True)
        return self._run_cmd(cmd, env, work_dir)

    def _run_cmd(
        self, cmd: list, env: dict, work_dir: pathlib.Path,
    ) -> tuple:
        try:
            proc = subprocess.run(
                cmd, env=env, cwd=str(work_dir),
                capture_output=True, text=True,
                timeout=self._target.seed_timeout_sec or None,
                check=False,
            )
            return proc.returncode, proc.stdout, proc.stderr, False
        except subprocess.TimeoutExpired as exc:
            so = exc.stdout or b""
            if isinstance(so, bytes):
                so = so.decode(errors="replace")
            se = exc.stderr or b""
            if isinstance(se, bytes):
                se = se.decode(errors="replace")
            return -1, so, se, True

    def _build_env(self, gcov_prefix: pathlib.Path, with_c_trace: bool) -> dict:
        env = os.environ.copy()
        env.update(self._target.extra_env)
        env["GCOV_PREFIX"] = str(gcov_prefix)
        if self._target.extra_pythonpath:
            extra = os.pathsep.join(str(p) for p in self._target.extra_pythonpath)
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                extra + os.pathsep + existing if existing else extra
            )
        env.pop("COVERAGE_FILE", None)
        if not with_c_trace:
            env.pop("MC_TRACE_FD", None)
        return env

    def cmd_repr(self, seed_path: pathlib.Path) -> str:
        if self._feedback_mode == "branch_state":
            cmd = [self._python, "-m", "polyfuzz.harness.seed_bootstrap",
                   str(seed_path), "<work>", "<scope>"]
        else:
            cmd = [self._python, "-m", "coverage", "run", "--branch",
                   f"--source={','.join(self._target.py_source_packages)}",
                   "--data-file=<auto>", str(seed_path)]
        return " ".join(shlex.quote(c) for c in cmd)


def _tail(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return "...[truncated]...\n" + s[-n:]
