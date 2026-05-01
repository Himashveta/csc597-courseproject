"""Python-side coverage collection.

We do NOT run coverage.py in-process. Each seed runs in a subprocess
(see harness.runner) and writes a per-seed SQLite database. This module
parses that database into a normalised set of (file, line) pairs and a
set of (file, branch_arc) pairs. We use coverage.py's own API where
possible to avoid getting tangled in schema versions.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import FrozenSet, Iterable, Set, Tuple

import coverage  # type: ignore[import-untyped]


LineKey = Tuple[str, int]                     # (filename, lineno)
ArcKey = Tuple[str, int, int]                 # (filename, src, dst)


@dataclasses.dataclass(frozen=True)
class PythonCoverageSnapshot:
    """A single seed's Python coverage."""
    lines: FrozenSet[LineKey]
    arcs: FrozenSet[ArcKey]

    def line_count(self) -> int:
        return len(self.lines)

    def arc_count(self) -> int:
        return len(self.arcs)


class PythonCoverage:
    """Reads coverage.py databases produced by `coverage run --branch`."""

    def __init__(
        self,
        include_packages: Iterable[str],
    ) -> None:
        # We translate package names into prefix matches against the
        # absolute paths recorded by coverage.py. This is more robust
        # than passing --include patterns to subprocess invocations,
        # because coverage.py records canonical absolute paths whose
        # exact form depends on how the target package was imported.
        self._packages = tuple(p for p in include_packages)

    def read(self, data_file: pathlib.Path) -> PythonCoverageSnapshot:
        """Read a single coverage.py db file and return a snapshot.

        Returns an empty snapshot if the file is missing — this can
        happen if a seed crashed before coverage.py had a chance to
        flush, in which case we just credit zero new Python coverage.
        """
        if not data_file.exists():
            return PythonCoverageSnapshot(frozenset(), frozenset())

        cov = coverage.Coverage(data_file=str(data_file))
        try:
            cov.load()
        except Exception:
            # Corrupt / partial database. Treat as empty rather than
            # propagating; the fuzzer should never fail because the
            # SUT crashed weirdly.
            return PythonCoverageSnapshot(frozenset(), frozenset())

        data = cov.get_data()

        lines: Set[LineKey] = set()
        arcs: Set[ArcKey] = set()

        for fname in data.measured_files():
            if not self._matches_scope(fname):
                continue
            file_lines = data.lines(fname) or []
            for ln in file_lines:
                lines.add((fname, ln))
            file_arcs = data.arcs(fname) or []
            for src, dst in file_arcs:
                arcs.add((fname, src, dst))

        return PythonCoverageSnapshot(frozenset(lines), frozenset(arcs))

    def _matches_scope(self, filename: str) -> bool:
        """True iff filename belongs to one of our target packages."""
        # coverage.py stores absolute paths. We pattern-match in two ways:
        #   1. As a package: "/<pkg/path/segment>/" — works for nested
        #      modules like torch._dynamo, torch._inductor, torch.fx.
        #   2. As a single-file module: "/<pkg_name>.py" — works for
        #      ad-hoc scripts and the mock target.
        if not self._packages:
            return True
        for pkg in self._packages:
            seg = "/" + pkg.replace(".", "/") + "/"
            if seg in filename:
                return True
            leaf = pkg.rsplit(".", 1)[-1]
            if filename.endswith("/" + leaf + ".py") \
                    or filename == leaf + ".py":
                return True
        return False
