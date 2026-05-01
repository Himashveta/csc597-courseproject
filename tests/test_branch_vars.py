"""Tests for branch-variable harvesting (Iteration 2)."""
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

import pytest

from polyfuzz.coverage.branch_state import BranchStateSet
from polyfuzz.coverage.branch_vars import (
    PythonBranchSnapshot,
    PythonBranchTracer,
    _pow2_bucket,
    _value_class,
    find_branch_lines,
    read_py_events,
)
from polyfuzz.coverage.c_branch_vars import (
    CBranchSnapshot,
    parse_trace_file,
)


# ---------------------------------------------------------------------------
# Static branch-line discovery
# ---------------------------------------------------------------------------

def test_find_branch_lines_picks_up_if_while_assert():
    src = (
        "def f(x, y):\n"
        "    if x > 0:\n"
        "        return x\n"
        "    while y < 100:\n"
        "        y += 1\n"
        "    assert y > 0\n"
        "    return y\n"
    )
    bl = find_branch_lines(src)
    # Lines: if at 2, while at 4, assert at 6.
    assert 2 in bl and "x" in bl[2]
    assert 4 in bl and "y" in bl[4]
    assert 6 in bl and "y" in bl[6]


def test_find_branch_lines_handles_attribute_chains():
    src = (
        "def f(t):\n"
        "    if t.shape[0] > 0 and t.dtype == 'fp32':\n"
        "        return t\n"
    )
    bl = find_branch_lines(src)
    assert 2 in bl
    # Should pick up dotted access, not just bare names.
    assert any("t.shape" in n or "t.dtype" in n for n in bl[2])


def test_find_branch_lines_returns_empty_on_syntax_error():
    assert find_branch_lines("def broken(:") == {}


# ---------------------------------------------------------------------------
# Value-class bucketing
# ---------------------------------------------------------------------------

def test_value_class_buckets_ints_by_sign_and_magnitude():
    assert _value_class(0) == "int:0"
    # 5 and 7 should bucket the same (both 2^2 range)
    assert _value_class(5) == _value_class(7)
    # But 5 and 200 should NOT bucket the same.
    assert _value_class(5) != _value_class(200)


def test_value_class_distinguishes_negative_from_positive():
    assert _value_class(-5) != _value_class(5)


def test_value_class_distinguishes_int_from_float_from_string():
    assert _value_class(0) != _value_class(0.0)
    assert _value_class(0) != _value_class("")


def test_value_class_buckets_collections_by_length():
    # Lengths 5 and 7 both fall in the 2^2 bucket (floor log2 of each is 2).
    assert _value_class([1, 2, 3, 4, 5]) == _value_class([1, 2, 3, 4, 5, 6, 7])
    # Length 5 and 100 do not.
    assert _value_class([1, 2, 3, 4, 5]) != _value_class(list(range(100)))


def test_pow2_bucket_handles_edges():
    assert _pow2_bucket(0) == "0"
    assert _pow2_bucket(1) == "2^0"
    assert _pow2_bucket(2) == "2^1"
    assert _pow2_bucket(3) == "2^1"
    assert _pow2_bucket(4) == "2^2"


# ---------------------------------------------------------------------------
# Runtime tracer (in-process)
# ---------------------------------------------------------------------------

def test_tracer_records_branch_events_for_in_scope_files(tmp_path: pathlib.Path):
    # Write a small target module.
    target = tmp_path / "demo_target.py"
    target.write_text(
        "def classify(x):\n"
        "    if x > 100:\n"
        "        return 'big'\n"
        "    if x == 0:\n"
        "        return 'zero'\n"
        "    return 'other'\n"
    )

    # Write a runner that imports the module and runs a few inputs.
    runner = tmp_path / "runner.py"
    runner.write_text(
        f"import sys, pathlib\n"
        f"sys.path.insert(0, str(pathlib.Path({str(target.parent)!r})))\n"
        f"sys.path.insert(0, {str(pathlib.Path(__file__).resolve().parents[1] / 'src')!r})\n"
        f"from polyfuzz.coverage.branch_vars import PythonBranchTracer\n"
        f"import demo_target\n"
        f"tracer = PythonBranchTracer(scope=['demo_target'])\n"
        f"tracer.install()\n"
        f"for v in [5, 200, 0, -1, 7]:\n"
        f"    demo_target.classify(v)\n"
        f"tracer.uninstall()\n"
        f"snap = tracer.snapshot()\n"
        f"import json\n"
        f"print(snap.event_count())\n"
    )
    proc = subprocess.run(
        [sys.executable, str(runner)],
        capture_output=True, text=True, check=True,
    )
    n = int(proc.stdout.strip())
    # We expect distinct value-class signatures per branch outcome.
    # x=5 and x=7 collapse (same bucket); x=200, x=0, x=-1 are distinct.
    # That's 4 unique signatures at line 2, plus signatures at line 4
    # (only reached when x<=100). Should be >= 4 and small.
    assert n >= 4
    assert n < 20


def test_read_py_events_returns_empty_on_missing_file(tmp_path):
    snap = read_py_events(tmp_path / "does_not_exist.json")
    assert snap.event_count() == 0


def test_read_py_events_recovers_well_formed_data(tmp_path):
    import json
    p = tmp_path / "events.json"
    p.write_text(json.dumps([
        ["foo.py", 10, "abc"],
        ["bar.py", 20, "def"],
    ]))
    snap = read_py_events(p)
    assert snap.event_count() == 2


def test_read_py_events_drops_malformed_rows(tmp_path):
    import json
    p = tmp_path / "events.json"
    p.write_text(json.dumps([
        ["foo.py", 10, "abc"],     # ok
        ["bar.py", "not-int", "x"],  # bad lineno
        "not-a-list",                # bad shape
    ]))
    snap = read_py_events(p)
    assert snap.event_count() == 1


# ---------------------------------------------------------------------------
# C trace parsing
# ---------------------------------------------------------------------------

def test_c_trace_parser_extracts_events(tmp_path):
    p = tmp_path / "trace.txt"
    p.write_text(
        "1\top->op_type=1\n"
        "2\top->rank=2\n"
        "3\top->shape[0]=64;op->shape[1]=64\n"
    )
    snap = parse_trace_file(p)
    assert snap.event_count() == 3


def test_c_trace_parser_buckets_so_similar_shapes_collapse(tmp_path):
    """64 and 65 should bucket the same (both 2^6); 64 and 128 differ."""
    p1 = tmp_path / "t1.txt"
    p1.write_text("3\tx=64;y=64\n")
    p2 = tmp_path / "t2.txt"
    p2.write_text("3\tx=65;y=65\n")
    p3 = tmp_path / "t3.txt"
    p3.write_text("3\tx=128;y=128\n")
    s1 = parse_trace_file(p1)
    s2 = parse_trace_file(p2)
    s3 = parse_trace_file(p3)
    assert s1.events == s2.events    # same bucket
    assert s1.events != s3.events    # different bucket


def test_c_trace_parser_returns_empty_on_missing(tmp_path):
    snap = parse_trace_file(tmp_path / "nope.txt")
    assert snap.event_count() == 0


def test_c_trace_parser_skips_garbage_lines(tmp_path):
    p = tmp_path / "trace.txt"
    p.write_text(
        "this is not a probe record\n"
        "1\tx=42\n"
        "\n"
        "another bogus line\n"
        "2\ty=10\n"
    )
    snap = parse_trace_file(p)
    assert snap.event_count() == 2


# ---------------------------------------------------------------------------
# BranchStateSet
# ---------------------------------------------------------------------------

def _empty_py_snap():
    return PythonBranchSnapshot(frozenset())


def _empty_c_snap():
    return CBranchSnapshot(frozenset())


def test_branch_state_empty_inputs_produce_no_novelty():
    bs = BranchStateSet()
    delta = bs.update(_empty_py_snap(), _empty_c_snap())
    assert not delta.is_novel()
    assert delta.total_new() == 0


def test_branch_state_first_observation_is_always_novel():
    bs = BranchStateSet()
    py = PythonBranchSnapshot(frozenset({("foo.py", 10, "abc")}))
    cc = CBranchSnapshot(frozenset({("c_probe", 1, "xyz")}))
    delta = bs.update(py, cc)
    assert delta.is_novel()
    assert delta.new_py_events == 1
    assert delta.new_c_events == 1


def test_branch_state_repeats_are_not_novel():
    bs = BranchStateSet()
    py = PythonBranchSnapshot(frozenset({("foo.py", 10, "abc")}))
    bs.update(py, _empty_c_snap())
    delta2 = bs.update(py, _empty_c_snap())
    assert not delta2.is_novel()


def test_branch_state_disjoint_python_and_c_each_register():
    bs = BranchStateSet()
    py = PythonBranchSnapshot(frozenset({("a.py", 1, "x")}))
    cc = CBranchSnapshot(frozenset({("c_probe", 1, "y")}))
    d1 = bs.update(py, _empty_c_snap())
    d2 = bs.update(_empty_py_snap(), cc)
    assert d1.is_novel() and d1.new_py_events == 1 and d1.new_c_events == 0
    assert d2.is_novel() and d2.new_py_events == 0 and d2.new_c_events == 1


def test_branch_state_summary_reflects_cumulative_inputs():
    bs = BranchStateSet()
    bs.update(
        PythonBranchSnapshot(frozenset({("a.py", 1, "x"), ("a.py", 2, "y")})),
        CBranchSnapshot(frozenset({("c_probe", 1, "z")})),
    )
    bs.update(
        PythonBranchSnapshot(frozenset({("a.py", 1, "x"), ("b.py", 5, "w")})),
        CBranchSnapshot(frozenset({("c_probe", 2, "v")})),
    )
    s = bs.coverage_summary()
    assert s["py_branch_events"] == 3
    assert s["c_branch_events"] == 2
    assert s["total_events"] == 5
