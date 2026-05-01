"""Python branch-variable tracing.

Real PolyFuzz instruments per-language frontends to harvest the
runtime values of variables that gate branches. For Python this is
straightforward via sys.settrace: the 'line' trace event fires on
every executed line, and we can inspect the calling frame's locals
at the moment of the branch predicate.

We narrow this in two ways to keep overhead tractable:
  1. We only trace files that match the target's py_source_packages
     scope (same predicate as PythonCoverage._matches_scope).
  2. We only emit a record at lines that are bytecode-level
     conditional branches. Static AST analysis up front identifies
     which lines those are; the tracer skips everything else.

The output is a list of BranchEvent records. Each record carries:
  - file:line of the branch
  - a hash of the local-variable value snapshot
  - a coarse "value class" for the most relevant variables (so seeds
    that take the same branch with similar shapes don't all look
    novel)

Trace overhead is significant (Python's settrace is ~10x slowdown
on covered code). For a fuzzer this is acceptable — coverage.py's
overhead is comparable, and we run subprocess-per-seed so the
slowdown only hits the seed, not the orchestrator.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import pathlib
import sys
from types import FrameType
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


# A BranchEvent identifies one execution of one branch with one
# value-class signature. Different value classes at the same branch
# are different events; same value class is the same event.
BranchEvent = Tuple[str, int, str]   # (filename, lineno, value_class_hash)


# --------------------------------------------------------------------------
# Static branch-line discovery
# --------------------------------------------------------------------------

def find_branch_lines(source: str) -> Dict[int, List[str]]:
    """Return {lineno: [variable_names_used_in_predicate, ...]}.

    For each If, While, IfExp, comprehension-with-if, and assert in
    the AST, record the line number and the names referenced in the
    predicate. These are the variables we want to harvest at runtime.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    result: Dict[int, List[str]] = {}

    class V(ast.NodeVisitor):
        def _record(self, lineno: int, predicate: ast.AST) -> None:
            names: List[str] = []
            for n in ast.walk(predicate):
                if isinstance(n, ast.Name):
                    names.append(n.id)
                elif isinstance(n, ast.Attribute):
                    # x.shape[0] -> "x.shape"
                    names.append(_attr_chain(n))
            if names:
                # Dedupe while preserving order.
                seen: Set[str] = set()
                uniq = [x for x in names if not (x in seen or seen.add(x))]
                result[lineno] = uniq

        def visit_If(self, node: ast.If) -> None:
            self._record(node.lineno, node.test)
            self.generic_visit(node)

        def visit_While(self, node: ast.While) -> None:
            self._record(node.lineno, node.test)
            self.generic_visit(node)

        def visit_IfExp(self, node: ast.IfExp) -> None:
            self._record(node.lineno, node.test)
            self.generic_visit(node)

        def visit_Assert(self, node: ast.Assert) -> None:
            self._record(node.lineno, node.test)
            self.generic_visit(node)

    V().visit(tree)
    return result


def _attr_chain(node: ast.Attribute) -> str:
    """Render an Attribute access chain back to a dotted string."""
    parts: List[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


# --------------------------------------------------------------------------
# Value-class bucketing
# --------------------------------------------------------------------------

def _value_class(v: Any) -> str:
    """Return a coarse equivalence class for a runtime value.

    The whole point of value-class bucketing is that fuzzer fitness
    should not credit a seed for hitting `if x > 0:` with x=37 if
    we already hit it with x=42. Both are in the "positive int" class.
    But x=-1 is a different class because the branch outcome differs.

    We bucket numerics by sign/zero/range; sequences by length bucket;
    types by category. Anything else falls back to type name. Bucket
    ranges are powers of two for log-uniform coverage.
    """
    if v is None:
        return "None"
    if isinstance(v, bool):
        return f"bool:{v}"
    if isinstance(v, int):
        if v == 0:
            return "int:0"
        if v < 0:
            return f"int:neg:{_pow2_bucket(-v)}"
        return f"int:pos:{_pow2_bucket(v)}"
    if isinstance(v, float):
        if v != v:           # NaN
            return "float:nan"
        if v == 0.0:
            return "float:0"
        if v < 0:
            return f"float:neg:{_pow2_bucket(int(-v) + 1)}"
        return f"float:pos:{_pow2_bucket(int(v) + 1)}"
    if isinstance(v, str):
        return f"str:len{_pow2_bucket(len(v))}"
    if isinstance(v, (list, tuple, set, dict)):
        return f"{type(v).__name__}:len{_pow2_bucket(len(v))}"
    # Catch tensor-like objects without importing torch:
    if hasattr(v, "shape") and hasattr(v, "dtype"):
        try:
            shape = tuple(int(d) for d in v.shape)
        except Exception:
            shape = None
        try:
            dtype = str(v.dtype)
        except Exception:
            dtype = "?"
        return f"tensor:{dtype}:rank{len(shape) if shape else '?'}"
    return f"obj:{type(v).__name__}"


def _pow2_bucket(n: int) -> str:
    """Return the floor-log2 bucket name for a non-negative integer."""
    if n <= 0:
        return "0"
    bits = 0
    while n > 1:
        n >>= 1
        bits += 1
    return f"2^{bits}"


# --------------------------------------------------------------------------
# Tracer
# --------------------------------------------------------------------------

@dataclasses.dataclass
class PythonBranchSnapshot:
    """Branch-variable events from one seed run."""
    events: FrozenSet[BranchEvent]

    def event_count(self) -> int:
        return len(self.events)


def read_py_events(path: pathlib.Path) -> PythonBranchSnapshot:
    """Load a py_events.json file written by seed_bootstrap.

    Returns an empty snapshot if the file is missing — typical when
    a seed crashes before atexit fires.
    """
    if not path.exists():
        return PythonBranchSnapshot(frozenset())
    try:
        import json
        with path.open() as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return PythonBranchSnapshot(frozenset())
    events: Set[BranchEvent] = set()
    for ev in raw:
        if isinstance(ev, list) and len(ev) == 3:
            try:
                events.add((str(ev[0]), int(ev[1]), str(ev[2])))
            except (TypeError, ValueError):
                continue
    return PythonBranchSnapshot(frozenset(events))


class PythonBranchTracer:
    """sys.settrace-based branch-variable harvester.

    Usage (in a seed subprocess):

        tracer = PythonBranchTracer(scope=["mock_compiler"])
        tracer.install()
        ... run target code ...
        tracer.uninstall()
        events = tracer.events()
        # Persist to a JSON file the parent process reads later.
    """

    def __init__(
        self,
        scope: Iterable[str],
        branch_lines_by_file: Optional[Dict[str, Dict[int, List[str]]]] = None,
    ) -> None:
        self._scope = tuple(scope)
        # branch_lines_by_file[abs_path][lineno] -> [var_names]. If None,
        # we discover branch lines lazily by parsing each new file we see.
        self._branch_map: Dict[str, Dict[int, List[str]]] = (
            branch_lines_by_file or {}
        )
        self._events: Set[BranchEvent] = set()
        self._installed = False

    # -- public ----------------------------------------------------

    def install(self) -> None:
        if self._installed:
            return
        sys.settrace(self._global_trace)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        sys.settrace(None)
        self._installed = False

    def snapshot(self) -> PythonBranchSnapshot:
        return PythonBranchSnapshot(frozenset(self._events))

    # -- trace internals -------------------------------------------

    def _global_trace(self, frame: FrameType, event: str, arg: Any):
        # Only trace files in scope. Returning None disables further
        # tracing for this frame.
        filename = frame.f_code.co_filename
        if not self._in_scope(filename):
            return None
        # Make sure we know the branch lines for this file.
        self._ensure_branch_map(filename)
        return self._local_trace

    def _local_trace(self, frame: FrameType, event: str, arg: Any):
        if event != "line":
            return self._local_trace
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        per_file = self._branch_map.get(filename)
        if per_file is None:
            return self._local_trace
        var_names = per_file.get(lineno)
        if not var_names:
            return self._local_trace
        # Compose value-class signature from the named locals.
        cls_parts: List[str] = []
        local_ns = frame.f_locals
        global_ns = frame.f_globals
        for name in var_names:
            v = self._lookup(name, local_ns, global_ns)
            cls_parts.append(f"{name}={_value_class(v)}")
        sig = ";".join(cls_parts)
        digest = hashlib.blake2b(sig.encode(), digest_size=8).hexdigest()
        self._events.add((filename, lineno, digest))
        return self._local_trace

    # -- helpers ---------------------------------------------------

    def _in_scope(self, filename: str) -> bool:
        for pkg in self._scope:
            seg = "/" + pkg.replace(".", "/") + "/"
            if seg in filename:
                return True
            leaf = pkg.rsplit(".", 1)[-1]
            if filename.endswith("/" + leaf + ".py") \
                    or filename.endswith(leaf + ".py"):
                return True
        return False

    def _ensure_branch_map(self, filename: str) -> None:
        if filename in self._branch_map:
            return
        try:
            src = pathlib.Path(filename).read_text()
        except OSError:
            self._branch_map[filename] = {}
            return
        self._branch_map[filename] = find_branch_lines(src)

    @staticmethod
    def _lookup(dotted: str, local_ns: dict, global_ns: dict) -> Any:
        """Resolve a dotted name in the given namespaces. Returns
        a sentinel string on lookup failure rather than raising."""
        parts = dotted.split(".")
        head = parts[0]
        if head in local_ns:
            v = local_ns[head]
        elif head in global_ns:
            v = global_ns[head]
        else:
            return "<missing>"
        for p in parts[1:]:
            try:
                v = getattr(v, p)
            except AttributeError:
                return "<missing>"
        return v
