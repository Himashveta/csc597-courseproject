"""Unified Python+C/C++ coverage bitmap.

This is the conceptual core of PolyFuzz. Both Python coverage (lines
and branch arcs from coverage.py) and C/C++ coverage (lines from gcov)
are mapped into a single AFL-style saturating-counter bitmap. The
bitmap drives:

  - Novelty detection: a seed is interesting iff it sets at least one
    new bit, regardless of whether the new bit comes from Python or
    C/C++.
  - Fitness: scored as a weighted sum of new-Python-lines,
    new-Python-arcs, new-C-lines, plus a flat bonus for any bug
    discovered.

The key design decision is keeping per-language deltas separately but
merging the *novelty signal* into one bitmap. That means a seed which
only adds Python coverage and a seed which only adds C coverage are
both retained, but a seed that re-runs the same Python+C path is not.
This is the property single-language fuzzers cannot achieve.

Hashing scheme:
  Python line:  blake2b(b"py_line|" + filename + ":" + lineno)[:8]
  Python arc :  blake2b(b"py_arc|"  + filename + ":" + src + ":" + dst)[:8]
  C line     :  blake2b(b"cc_line|" + filename + ":" + lineno)[:8]

The 8-byte hash is reduced modulo bitmap_size to get a slot index.
Each slot is a saturating uint8 counter.
"""

from __future__ import annotations

import dataclasses
import hashlib
import struct
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from polyfuzz.coverage.python_cov import PythonCoverageSnapshot
    from polyfuzz.coverage.cpp_cov import CppCoverageSnapshot


@dataclasses.dataclass(frozen=True)
class CoverageDelta:
    """How much new coverage a single seed produced."""
    new_py_lines: int
    new_py_arcs: int
    new_cc_lines: int
    new_cc_branches: int
    new_bitmap_bits: int

    def total_new(self) -> int:
        return (self.new_py_lines + self.new_py_arcs
                + self.new_cc_lines + self.new_cc_branches)

    def is_novel(self) -> bool:
        return self.new_bitmap_bits > 0


class UnifiedBitmap:
    """An AFL-style bitmap fused across the Python/C boundary."""

    DEFAULT_SIZE = 65_536

    def __init__(self, size: int = DEFAULT_SIZE) -> None:
        if size <= 0 or (size & (size - 1)) != 0:
            raise ValueError(f"bitmap size must be a power of 2, got {size}")
        self._size = size
        self._mask = size - 1
        self._buf = bytearray(size)
        self._py_lines_total: set[Tuple[str, int]] = set()
        self._py_arcs_total: set[Tuple[str, int, int]] = set()
        self._cc_lines_total: set[Tuple[str, int]] = set()
        self._cc_branches_total: set[Tuple[str, int, int]] = set()

    @property
    def size(self) -> int:
        return self._size

    def coverage_summary(self) -> dict:
        nz = sum(1 for b in self._buf if b)
        return {
            "bitmap_size": self._size,
            "bitmap_bits_set": nz,
            "bitmap_fill_pct": 100.0 * nz / self._size,
            "py_lines": len(self._py_lines_total),
            "py_arcs": len(self._py_arcs_total),
            "cc_lines": len(self._cc_lines_total),
            "cc_branches": len(self._cc_branches_total),
        }

    def update(
        self,
        py: "PythonCoverageSnapshot",
        cc: "CppCoverageSnapshot",
    ) -> CoverageDelta:
        new_py_lines = 0
        new_py_arcs = 0
        new_cc_lines = 0
        new_cc_branches = 0
        new_bits = 0

        for line_key in py.lines:
            if line_key not in self._py_lines_total:
                self._py_lines_total.add(line_key)
                new_py_lines += 1
            if self._set_slot(self._hash_py_line(line_key)):
                new_bits += 1

        for arc_key in py.arcs:
            if arc_key not in self._py_arcs_total:
                self._py_arcs_total.add(arc_key)
                new_py_arcs += 1
            if self._set_slot(self._hash_py_arc(arc_key)):
                new_bits += 1

        for line_key in cc.lines:
            if line_key not in self._cc_lines_total:
                self._cc_lines_total.add(line_key)
                new_cc_lines += 1
            if self._set_slot(self._hash_cc_line(line_key)):
                new_bits += 1

        for br_key in cc.branches:
            if br_key not in self._cc_branches_total:
                self._cc_branches_total.add(br_key)
                new_cc_branches += 1
            if self._set_slot(self._hash_cc_branch(br_key)):
                new_bits += 1

        return CoverageDelta(
            new_py_lines=new_py_lines,
            new_py_arcs=new_py_arcs,
            new_cc_lines=new_cc_lines,
            new_cc_branches=new_cc_branches,
            new_bitmap_bits=new_bits,
        )

    # -- internal -------------------------------------------------

    def _set_slot(self, idx: int) -> bool:
        was_zero = self._buf[idx] == 0
        if self._buf[idx] < 0xFF:
            self._buf[idx] += 1
        return was_zero

    def _hash_py_line(self, key: Tuple[str, int]) -> int:
        return self._h(b"py_line|", f"{key[0]}:{key[1]}".encode())

    def _hash_py_arc(self, key: Tuple[str, int, int]) -> int:
        return self._h(b"py_arc|",
                       f"{key[0]}:{key[1]}:{key[2]}".encode())

    def _hash_cc_line(self, key: Tuple[str, int]) -> int:
        return self._h(b"cc_line|", f"{key[0]}:{key[1]}".encode())

    def _hash_cc_branch(self, key: Tuple[str, int, int]) -> int:
        return self._h(b"cc_branch|",
                       f"{key[0]}:{key[1]}:{key[2]}".encode())

    def _h(self, tag: bytes, payload: bytes) -> int:
        h = hashlib.blake2b(tag + payload, digest_size=8).digest()
        (val,) = struct.unpack("<Q", h)
        return val & self._mask
