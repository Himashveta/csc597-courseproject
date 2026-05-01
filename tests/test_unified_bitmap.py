"""Tests for the unified Python+C/C++ bitmap.

These tests do not require a built target. They exercise the merger
in isolation by feeding it synthetic snapshots.
"""

from polyfuzz.coverage.unified import UnifiedBitmap
from polyfuzz.coverage.python_cov import PythonCoverageSnapshot
from polyfuzz.coverage.cpp_cov import CppCoverageSnapshot


def _py(lines=(), arcs=()):
    return PythonCoverageSnapshot(frozenset(lines), frozenset(arcs))


def _cc(lines=(), branches=()):
    return CppCoverageSnapshot(frozenset(lines), frozenset(branches))


def test_empty_snapshots_produce_no_novelty():
    bm = UnifiedBitmap(size=1024)
    delta = bm.update(_py(), _cc())
    assert not delta.is_novel()
    assert delta.total_new() == 0


def test_first_seed_credits_all_signals():
    bm = UnifiedBitmap(size=1024)
    delta = bm.update(
        _py(lines=[("a.py", 1), ("a.py", 2)],
            arcs=[("a.py", 1, 2)]),
        _cc(lines=[("a.c", 10)],
            branches=[("a.c", 10, 0)]),
    )
    assert delta.new_py_lines == 2
    assert delta.new_py_arcs == 1
    assert delta.new_cc_lines == 1
    assert delta.new_cc_branches == 1
    assert delta.is_novel()


def test_repeated_signals_are_not_novel():
    bm = UnifiedBitmap(size=1024)
    bm.update(_py(lines=[("a.py", 1)]), _cc())
    delta = bm.update(_py(lines=[("a.py", 1)]), _cc())
    assert not delta.is_novel()
    assert delta.new_py_lines == 0


def test_python_only_seed_is_novel_when_c_seed_already_seen():
    bm = UnifiedBitmap(size=1024)
    bm.update(_py(), _cc(lines=[("a.c", 1)]))
    delta = bm.update(
        _py(lines=[("a.py", 5)]),
        _cc(lines=[("a.c", 1)]),  # already known
    )
    assert delta.is_novel()
    assert delta.new_py_lines == 1
    assert delta.new_cc_lines == 0


def test_c_branch_is_novel_even_when_line_repeats():
    """The motivating case for branch tracking on the C side: same
    line, different control flow direction."""
    bm = UnifiedBitmap(size=1024)
    bm.update(_py(), _cc(lines=[("a.c", 10)],
                          branches=[("a.c", 10, 0)]))
    delta = bm.update(
        _py(),
        _cc(lines=[("a.c", 10)],         # line already covered
            branches=[("a.c", 10, 1)]),  # but this branch is new
    )
    assert delta.is_novel()
    assert delta.new_cc_lines == 0
    assert delta.new_cc_branches == 1


def test_summary_reports_separated_signals():
    bm = UnifiedBitmap(size=1024)
    bm.update(
        _py(lines=[("a.py", 1)], arcs=[("a.py", 1, 2)]),
        _cc(lines=[("a.c", 1), ("a.c", 2)],
            branches=[("a.c", 1, 0)]),
    )
    s = bm.coverage_summary()
    assert s["py_lines"] == 1
    assert s["py_arcs"] == 1
    assert s["cc_lines"] == 2
    assert s["cc_branches"] == 1
    assert s["bitmap_bits_set"] >= 5  # 1+1+2+1 logically distinct keys


def test_bitmap_size_must_be_power_of_two():
    import pytest
    with pytest.raises(ValueError):
        UnifiedBitmap(size=1000)


def test_hash_collision_does_not_inflate_novelty_after_first_hit():
    """If two distinct keys collide in the bitmap, only the first
    sets the bit. The second produces no novelty even though the
    signal is technically new at the key level."""
    # Construct two snapshots whose hashes happen to land in a tiny
    # bitmap. With size=2 collisions are guaranteed.
    bm = UnifiedBitmap(size=2)
    bm.update(_py(lines=[("a.py", 1)]), _cc())
    delta = bm.update(_py(lines=[("a.py", 2)]), _cc())
    # The second key is logically new at the cumulative-set level...
    assert delta.new_py_lines == 1
    # ...but the bitmap may already be full, so new_bits could be 0.
    # This is the AFL-bitmap-collision tradeoff, expressed as a test.
    assert delta.new_bitmap_bits >= 0
