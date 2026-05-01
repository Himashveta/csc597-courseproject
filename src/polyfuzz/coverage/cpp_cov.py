"""C/C++ coverage collection via gcov.

The instrumented C target writes .gcno files at compile time and
.gcda files at process exit (or when __gcov_dump is called). We
rely on the canonical exit-time flush, which is automatic for any
process that terminates cleanly via exit() / return-from-main.

For each seed run we:
  1. Set GCOV_PREFIX to a per-seed directory before launching the
     subprocess. gcov writes .gcda files under that prefix, mirroring
     the absolute path of the corresponding .gcno but rooted at the
     prefix dir.
  2. After the subprocess exits, locate the .gcda files, find the
     matching .gcno files in the gcov source tree, and run gcov in
     JSON intermediate mode with branch info (`gcov -j -b`).
  3. Decompress the resulting .gcov.json.gz and extract:
       - executed lines (count > 0)
       - taken branches (per-line branch_idx)

We keep both signals distinct: branch-level coverage is the one that
discriminates between seeds that visit the same lines via different
control-flow paths, but line-level coverage is a useful sanity check
and shows up in the public summary.

The user can override this with their own parser by subclassing
CppCoverage if their toolchain doesn't support gcov -j (gcov < 9).
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import pathlib
import subprocess
import tempfile
from typing import FrozenSet, Iterable, List, Optional, Set, Tuple


LineKey = Tuple[str, int]              # (source_filename, lineno)
BranchKey = Tuple[str, int, int]       # (source_filename, lineno, branch_idx)


@dataclasses.dataclass(frozen=True)
class CppCoverageSnapshot:
    """A single seed's C/C++ coverage."""
    lines: FrozenSet[LineKey]
    branches: FrozenSet[BranchKey]

    def line_count(self) -> int:
        return len(self.lines)

    def branch_count(self) -> int:
        return len(self.branches)


class CppCoverage:
    """Walks .gcda files written under GCOV_PREFIX and parses gcov JSON output."""

    def __init__(
        self,
        gcov_source_root: pathlib.Path,
        include_prefixes: Iterable[str],
        exclude_prefixes: Iterable[str] = (),
        gcov_bin: str = "gcov",
    ) -> None:
        self._gcov_source_root = gcov_source_root.resolve()
        self._include_prefixes = tuple(include_prefixes)
        self._exclude_prefixes = tuple(exclude_prefixes)
        self._gcov_bin = gcov_bin

    def read(self, gcov_prefix_dir: pathlib.Path) -> CppCoverageSnapshot:
        """Read coverage for one seed run."""
        if not gcov_prefix_dir.exists():
            return CppCoverageSnapshot(frozenset(), frozenset())

        gcda_files = list(gcov_prefix_dir.rglob("*.gcda"))
        if not gcda_files:
            return CppCoverageSnapshot(frozenset(), frozenset())

        lines: Set[LineKey] = set()
        branches: Set[BranchKey] = set()
        for gcda in gcda_files:
            ln_set, br_set = self._gcov_one(gcda)
            for fname, lno in ln_set:
                if not self._matches_scope(fname):
                    continue
                lines.add((fname, lno))
            for fname, lno, idx in br_set:
                if not self._matches_scope(fname):
                    continue
                branches.add((fname, lno, idx))
        return CppCoverageSnapshot(frozenset(lines), frozenset(branches))

    # ----------------------------------------------------------------

    def _gcov_one(
        self, gcda_path: pathlib.Path,
    ) -> Tuple[List[LineKey], List[BranchKey]]:
        """Run `gcov -j -b` on one .gcda and parse the resulting JSON.

        gcov requires the .gcno (compile-time notes) to be co-located
        with the .gcda. Under GCOV_PREFIX the .gcda lives in a
        per-seed mirror directory; the .gcno is back at the original
        build root. We bridge by symlinking the .gcno next to the
        .gcda before invoking gcov. The `-o` flag does NOT change
        where gcov looks for the .gcno — it only affects source-file
        lookup — so we cannot rely on it.

        We resolve gcda_path to an absolute path before invocation
        because gcov interprets relative paths against its own cwd
        (the temp dir we use for output), not against ours.
        """
        gcda_path = gcda_path.resolve()
        gcno_source = self._find_gcno_for(gcda_path)
        if gcno_source is None:
            return [], []

        gcno_link = gcda_path.with_suffix(".gcno")
        try:
            if gcno_link.exists() or gcno_link.is_symlink():
                gcno_link.unlink()
            gcno_link.symlink_to(gcno_source)
        except OSError:
            return [], []

        try:
            with tempfile.TemporaryDirectory() as work:
                work_path = pathlib.Path(work)
                cmd = [
                    self._gcov_bin,
                    "-j",         # JSON intermediate format
                    "-b",         # include branch counts
                    str(gcda_path),
                ]
                try:
                    subprocess.run(
                        cmd, cwd=str(work_path),
                        capture_output=True, timeout=15, check=False,
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    return [], []
                return self._parse_gcov_json_dir(work_path)
        finally:
            try:
                if gcno_link.is_symlink() or gcno_link.exists():
                    gcno_link.unlink()
            except OSError:
                pass

    def _parse_gcov_json_dir(
        self, dir_: pathlib.Path,
    ) -> Tuple[List[LineKey], List[BranchKey]]:
        lines: List[LineKey] = []
        branches: List[BranchKey] = []
        for path in list(dir_.glob("*.gcov.json.gz")) + list(dir_.glob("*.gcov.json")):
            try:
                if path.suffix == ".gz":
                    with gzip.open(path, "rt") as f:
                        obj = json.load(f)
                else:
                    with path.open() as f:
                        obj = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            ln, br = self._extract_from_obj(obj)
            lines.extend(ln)
            branches.extend(br)
        return lines, branches

    def _extract_from_obj(
        self, obj: dict,
    ) -> Tuple[List[LineKey], List[BranchKey]]:
        lines: List[LineKey] = []
        branches: List[BranchKey] = []
        for f in obj.get("files", []):
            fname = f.get("file", "")
            if not fname.startswith("/"):
                fname = str((self._gcov_source_root / fname).resolve())
            for ln in f.get("lines", []):
                lno = int(ln.get("line_number", 0))
                if ln.get("count", 0) > 0:
                    lines.append((fname, lno))
                for idx, br in enumerate(ln.get("branches", [])):
                    if br.get("count", 0) > 0:
                        branches.append((fname, lno, idx))
        return lines, branches

    def _find_gcno_for(self, gcda_path: pathlib.Path) -> Optional[pathlib.Path]:
        """Locate the .gcno file that pairs with gcda_path.

        Strategy: search the gcov source root for a .gcno with the same
        stem as the .gcda. Robust to GCOV_PREFIX rewriting because we
        only rely on the file basename, not on path correspondence.
        """
        stem = gcda_path.stem
        for gcno in self._gcov_source_root.rglob(stem + ".gcno"):
            return gcno
        return None

    def _matches_scope(self, filename: str) -> bool:
        if self._include_prefixes:
            ok = False
            for p in self._include_prefixes:
                if filename.startswith(p) or (("/" + p) in filename):
                    ok = True
                    break
            if not ok:
                return False
        for excl in self._exclude_prefixes:
            if filename.startswith(excl) or (("/" + excl) in filename):
                return False
        return True
