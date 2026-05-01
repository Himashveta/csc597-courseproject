"""Branch-state feedback (PolyFuzz Iteration 2).

Iteration 1 of this fuzzer used a hashed AFL-style bitmap fed by line
and edge coverage from both Python and C. That signal works but is
coarse: a seed that visits the same lines via different runtime
*values* — different shapes, different dtypes — looks identical.

PolyFuzz (Li et al., USENIX Sec '23) argues that the right signal is
*branch-variable values*: harvest, at every branch, the runtime values
of the variables in the predicate. A seed is novel when it produces
a (branch, value-class) pair we haven't seen.

This module is the unified-set analog of UnifiedBitmap:
  - PythonBranchSnapshot contributes events keyed by (file, line, value_hash)
  - CBranchSnapshot contributes events keyed by ("c_probe", branch_id, value_hash)
  - We union into one set; novelty = the union grew.

The bucketing is what makes this useful — without it, seeds with
matmul shape [64,64] and [65,65] would produce different events
even though they take the same control-flow paths. With log-uniform
power-of-two bucketing, "same outcome" maps to "same event" while
"different outcome" maps to "different event".
"""
from __future__ import annotations

import dataclasses
from typing import Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from polyfuzz.coverage.branch_vars import PythonBranchSnapshot, BranchEvent
    from polyfuzz.coverage.c_branch_vars import CBranchSnapshot


@dataclasses.dataclass(frozen=True)
class BranchStateDelta:
    """How much new branch-state a single seed produced."""
    new_py_events: int
    new_c_events: int

    def total_new(self) -> int:
        return self.new_py_events + self.new_c_events

    def is_novel(self) -> bool:
        return self.total_new() > 0


class BranchStateSet:
    """Cumulative set of (branch, value-class) events seen so far."""

    def __init__(self) -> None:
        self._py_events: Set = set()
        self._c_events: Set = set()

    def update(
        self,
        py: "PythonBranchSnapshot",
        cc: "CBranchSnapshot",
    ) -> BranchStateDelta:
        new_py = 0
        new_c = 0
        for ev in py.events:
            if ev not in self._py_events:
                self._py_events.add(ev)
                new_py += 1
        for ev in cc.events:
            if ev not in self._c_events:
                self._c_events.add(ev)
                new_c += 1
        return BranchStateDelta(new_py_events=new_py, new_c_events=new_c)

    def coverage_summary(self) -> dict:
        return {
            "py_branch_events": len(self._py_events),
            "c_branch_events": len(self._c_events),
            "total_events": len(self._py_events) + len(self._c_events),
        }
