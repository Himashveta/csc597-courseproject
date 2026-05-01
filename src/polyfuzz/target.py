"""Target specification.

A TargetSpec captures everything PolyFuzz needs to know about *where*
to look for coverage and *how* to invoke the system under test:

  - Which Python packages to instrument with coverage.py
  - Where the gcov .gcno/.gcda files live (the gcov source root)
  - Which file-path prefixes to include / exclude when reading those
    coverage signals
  - Extra environment variables required to make the build emit
    coverage at runtime (e.g. GCOV_PREFIX)

Two stock specs are provided:
  - MOCK_TARGET: ships with the artefact, builds in seconds
  - PYTORCH_TARGET: assumes a coverage-instrumented PyTorch build per
    the dl-compiler-fuzzing skill conventions.

A user can construct a custom TargetSpec for any other Python+C system.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Dict, List, Optional


@dataclasses.dataclass
class TargetSpec:
    name: str

    # Python coverage scope: passed to `coverage run --source=...`.
    py_source_packages: List[str]

    # C/C++ coverage scope: directory tree where .gcno files live and
    # path-prefix filters applied to gcov output.
    gcov_source_root: pathlib.Path
    gcov_include_prefixes: List[str]
    gcov_exclude_prefixes: List[str] = dataclasses.field(default_factory=list)

    # Extra env merged into the subprocess environment for each seed.
    extra_env: Dict[str, str] = dataclasses.field(default_factory=dict)

    # Optional per-seed timeout (seconds). 0 = no timeout.
    seed_timeout_sec: float = 30.0

    # Optional: extra Python path entries (e.g. so seeds can import
    # the target package from a non-standard install).
    extra_pythonpath: List[pathlib.Path] = dataclasses.field(default_factory=list)

    def sanity_check(self) -> Optional[str]:
        """Return None if the spec looks usable, else an error string."""
        if not self.gcov_source_root.exists():
            return f"gcov_source_root does not exist: {self.gcov_source_root}"
        return None


# ---------------------------------------------------------------------------
# Stock targets.
# ---------------------------------------------------------------------------

# Repo root, derived from this file's location, so the mock spec works
# without environment configuration.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


MOCK_TARGET = TargetSpec(
    name="mock",
    py_source_packages=["mock_compiler"],
    gcov_source_root=_REPO_ROOT / "target",
    gcov_include_prefixes=[str(_REPO_ROOT / "target")],
    gcov_exclude_prefixes=[],
    extra_env={
        # Default PYTHONPATH addition; PolyFuzz appends, doesn't overwrite.
    },
    extra_pythonpath=[_REPO_ROOT / "target"],
    seed_timeout_sec=10.0,
)


PYTORCH_TARGET = TargetSpec(
    name="pytorch",
    py_source_packages=[
        "torch._dynamo",
        "torch._inductor",
        "torch.fx",
    ],
    # The gcov source root is wherever the user built PyTorch from.
    gcov_source_root=pathlib.Path(
        os.environ.get("TORCH_ROOT", str(pathlib.Path.home() / "pytorch-ptcov"))
    ),
    gcov_include_prefixes=[
        "aten/",
        "torch/csrc/",
        "torch/csrc/inductor/",
    ],
    gcov_exclude_prefixes=[
        "third_party/",
        "build/",
        "test/",
    ],
    extra_env={
        # gcov dumps under here so seed runs don't trample each other.
        # PolyFuzz sets a per-seed subdirectory automatically.
        "GCOV_PREFIX_STRIP": "10",
        # ASAN/UBSan should be on the *bug-finding* build only; per the
        # skill, never mix gcov and ASAN. We keep these here as a
        # reminder; the user can override.
        "ASAN_OPTIONS": "abort_on_error=1,allocator_may_return_null=1",
        "UBSAN_OPTIONS": "print_stacktrace=1,halt_on_error=1",
    },
    seed_timeout_sec=60.0,
)
