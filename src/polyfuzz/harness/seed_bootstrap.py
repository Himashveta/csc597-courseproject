"""Bootstrap script run inside each seed subprocess.

The harness invokes:
    python -m polyfuzz.harness.seed_bootstrap <seed_file> <out_dir> <scope_csv>

Bootstrap responsibilities:
  1. Open a probe file at out_dir/c_trace.txt and dup it to fd 5,
     export MC_TRACE_FD=5 so the C target writes to it.
  2. Install PythonBranchTracer scoped to <scope_csv>.
  3. Run the seed file via runpy.run_path so it executes as if it
     were `python <seed_file>`.
  4. Uninstall the tracer and dump events to out_dir/py_events.json.
  5. Exit with the seed's natural exit code.

Errors during bootstrap itself are reported with exit code 99 to
distinguish them from real seed failures.

We do NOT wrap the seed in coverage.py here. The harness invokes
coverage.py separately for the Iteration 1 coverage signal, then
this bootstrap separately for the Iteration 2 branch-state signal.
A future optimisation would do both in one subprocess; for clarity
we keep them separate.
"""
from __future__ import annotations

import json
import os
import pathlib
import runpy
import sys
import traceback


def main() -> int:
    if len(sys.argv) < 4:
        sys.stderr.write(
            "usage: seed_bootstrap <seed_file> <out_dir> <scope_csv>\n"
        )
        return 99

    seed_file = pathlib.Path(sys.argv[1]).resolve()
    out_dir = pathlib.Path(sys.argv[2]).resolve()
    scope = [s for s in sys.argv[3].split(",") if s]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # C-side: open the probe file, dup to fd 5, set MC_TRACE_FD=5.
    # ------------------------------------------------------------------
    c_trace_path = out_dir / "c_trace.txt"
    try:
        # O_CLOEXEC=False because we want the fd to survive across
        # any further forks the seed itself might do (rare but defensive).
        raw_fd = os.open(c_trace_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        target_fd = 5
        if raw_fd != target_fd:
            os.dup2(raw_fd, target_fd)
            os.close(raw_fd)
        os.environ["MC_TRACE_FD"] = str(target_fd)
    except OSError as e:
        sys.stderr.write(f"bootstrap: failed to open C trace file: {e}\n")
        return 99

    # ------------------------------------------------------------------
    # Python-side: install branch tracer scoped to <scope>.
    # ------------------------------------------------------------------
    try:
        from polyfuzz.coverage.branch_vars import PythonBranchTracer
    except ImportError as e:
        sys.stderr.write(f"bootstrap: cannot import tracer: {e}\n")
        return 99

    tracer = PythonBranchTracer(scope=scope)
    py_events_path = out_dir / "py_events.json"

    # We register a finalizer to dump events even if the seed exits via
    # SystemExit, raises, or crashes via signal in the C library. atexit
    # doesn't fire on SIGSEGV, but it does fire on SystemExit and on
    # Python-level exceptions, which covers the common cases. Crashes
    # leave py_events.json empty, and the harness handles that.
    import atexit

    def _dump():
        try:
            tracer.uninstall()
            snap = tracer.snapshot()
            # Write JSON list of [filename, lineno, value_hash] triples.
            with py_events_path.open("w") as f:
                json.dump([list(ev) for ev in sorted(snap.events)], f)
        except Exception:
            # Last-ditch: don't let teardown errors mask the seed's exit.
            pass

    atexit.register(_dump)

    # ------------------------------------------------------------------
    # Run the seed.
    # ------------------------------------------------------------------
    tracer.install()
    try:
        # Use runpy so __name__ == '__main__' inside the seed.
        # Pass argv so the seed sees only its own filename.
        sys.argv = [str(seed_file)]
        runpy.run_path(str(seed_file), run_name="__main__")
    except SystemExit as e:
        # Propagate the exit code unchanged.
        return int(e.code) if isinstance(e.code, int) else (1 if e.code else 0)
    except BaseException:
        # Print the traceback to stderr (so the oracle can classify) and
        # return non-zero. Use 1 by convention.
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
