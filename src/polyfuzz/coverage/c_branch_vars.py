"""C/C++ branch-variable harvest reader.

The instrumented C target writes records of the form

    <branch_id>\t<name>=<val>;<name>=<val>;...\n

to a file descriptor named by MC_TRACE_FD. After a seed run we read
the file and bucket each record into a coarse value-class signature,
matching the bucketing scheme used by branch_vars.py for Python.

Each (branch_id, value_class_hash) pair is one BranchEvent, the same
shape as Python's branch events. This lets the unified bitmap (or the
new branch-state set) treat C and Python events uniformly.

Limitations carried forward to the report:
  - The C target must be source-instrumented with MC_PROBE macros.
    For the mock target this is fine; for PyTorch this would mean
    patching torch source. We document the obstacle.
  - Probe payloads are restricted to integer values rendered by
    snprintf. Float and tensor-shape probes would need additional
    macros; we don't implement them here because the mock target
    only branches on int-valued predicates.
"""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import re
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple


# Same shape as Python's BranchEvent: (branch_id_str, "", value_class_hash).
# The middle slot is empty so we can interleave Python and C events in
# the same set without collision (the prefix distinguishes them).
BranchEvent = Tuple[str, int, str]


@dataclasses.dataclass(frozen=True)
class CBranchSnapshot:
    events: FrozenSet[BranchEvent]

    def event_count(self) -> int:
        return len(self.events)


_RECORD_RE = re.compile(r"^(\d+)\t(.*)$")
_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_\->\.\[\]]*)=(-?\d+)")


def _value_class_int(v: int) -> str:
    """Same bucketing as Python's _value_class for ints."""
    if v == 0:
        return "int:0"
    if v < 0:
        return f"int:neg:{_pow2_bucket(-v)}"
    return f"int:pos:{_pow2_bucket(v)}"


def _pow2_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    bits = 0
    while n > 1:
        n >>= 1
        bits += 1
    return f"2^{bits}"


def parse_trace_file(path: pathlib.Path) -> CBranchSnapshot:
    """Read a probe trace file and return a CBranchSnapshot.

    Returns an empty snapshot if the file is missing or corrupt.
    """
    if not path.exists():
        return CBranchSnapshot(frozenset())

    events: Set[BranchEvent] = set()
    try:
        with path.open() as f:
            for line in f:
                ev = _parse_one(line)
                if ev is not None:
                    events.add(ev)
    except OSError:
        return CBranchSnapshot(frozenset(events))
    return CBranchSnapshot(frozenset(events))


def _parse_one(line: str) -> BranchEvent | None:
    """Parse one probe record into a BranchEvent."""
    m = _RECORD_RE.match(line.rstrip("\n"))
    if not m:
        return None
    branch_id = m.group(1)
    payload = m.group(2)

    # Bucket each (name=value) pair. Using a list of strings preserves
    # name order, which we need so that the same predicate at the same
    # branch always hashes to the same signature.
    parts: List[str] = []
    for kv in _KV_RE.finditer(payload):
        name = kv.group(1)
        try:
            val = int(kv.group(2))
        except ValueError:
            continue
        parts.append(f"{name}={_value_class_int(val)}")
    if not parts:
        return None
    sig = ";".join(parts)
    digest = hashlib.blake2b(
        ("c|" + branch_id + "|" + sig).encode(), digest_size=8
    ).hexdigest()
    # Use a synthetic "filename" of "c_probe" so events are
    # distinguishable from python events but uniform in shape.
    return ("c_probe", int(branch_id), digest)
