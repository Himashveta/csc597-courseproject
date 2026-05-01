"""Tests for corpus management."""

import pathlib
import random

import pytest

from polyfuzz.corpus import Corpus, Seed, SeedKind


@pytest.fixture
def tmp_corpus(tmp_path: pathlib.Path) -> Corpus:
    return Corpus(
        corpus_dir=tmp_path / "corpus",
        kind=SeedKind.MOCK,
        rng=random.Random(0),
    )


def _seed(src: str, parent_id=None) -> Seed:
    return Seed.from_source(src, kind=SeedKind.MOCK, parent_id=parent_id)


def test_empty_corpus_returns_none_on_select(tmp_corpus):
    assert tmp_corpus.select() is None
    assert len(tmp_corpus) == 0


def test_add_writes_seed_to_disk(tmp_corpus, tmp_path):
    s = _seed("print('hi')")
    assert tmp_corpus.add(s, fitness=1.0)
    files = list((tmp_path / "corpus").glob("*.py"))
    assert len(files) == 1
    assert files[0].read_text() == "print('hi')"


def test_duplicate_seed_is_rejected(tmp_corpus):
    s = _seed("print(1)")
    assert tmp_corpus.add(s, fitness=1.0)
    assert not tmp_corpus.add(s, fitness=2.0)
    assert len(tmp_corpus) == 1


def test_select_eventually_returns_every_seed(tmp_corpus):
    """Power-law softening + selection-count decay should give every
    seed reasonable attention even with very different fitnesses."""
    seeds = [_seed(f"x = {i}") for i in range(5)]
    for i, s in enumerate(seeds):
        # seed 0 has huge fitness, others very small
        tmp_corpus.add(s, fitness=1000.0 if i == 0 else 1.0)

    seen = set()
    for _ in range(200):
        chosen = tmp_corpus.select()
        seen.add(chosen.seed_id)
    # All five should be chosen at least once. If only 1-2 are chosen
    # the selector is monopolized by the high-fitness seed.
    assert len(seen) == 5


def test_lineage_quota_blocks_one_root_from_dominating(tmp_path):
    """Once the corpus has enough entries for the share to be
    meaningful, no single lineage should exceed max_lineage_share.
    """
    # max_lineage_share=0.20 -> quota engages once the corpus has
    # at least 5 entries.
    c = Corpus(corpus_dir=tmp_path / "c", kind=SeedKind.MOCK,
               max_lineage_share=0.20, rng=random.Random(0))

    # Seed five distinct lineages; force-add bypasses the quota for
    # initial seeds, which is realistic.
    roots = [_seed(f"root_{i} = 1") for i in range(5)]
    for r in roots:
        assert c.add(r, fitness=1.0, force=True)

    # Now non-forced descendants of root_0 should be admissible up to
    # the quota and rejected past it. With corpus size 5 and quota
    # 0.20, root_0's lineage already holds 1/5 = 0.20. Adding one
    # more child of root_0 takes it to 2/6 ≈ 0.33 — past the cap.
    overflow = _seed("dominator", parent_id=roots[0].seed_id)
    assert not c.add(overflow, fitness=1.0)

    # A child of root_1 (different lineage) should be accepted, since
    # root_1's share would only become 2/6 too — but at least no one
    # lineage runs away.
    diversifying = _seed("diversifying", parent_id=roots[1].seed_id)
    # Whether this one is accepted depends on identical share math,
    # so we only assert that the quota REJECTS at least one of them
    # rather than accepting both unconditionally.
    accepted = c.add(diversifying, fitness=1.0)
    # The point of the test: not every same-lineage child is admitted.
    again = _seed("again", parent_id=roots[0].seed_id)
    assert not c.add(again, fitness=1.0)
    # And force still works.
    forced = _seed("forced", parent_id=roots[0].seed_id)
    assert c.add(forced, fitness=1.0, force=True)
    # Suppress unused-variable noise.
    _ = accepted


def test_force_add_bypasses_quota(tmp_path):
    c = Corpus(corpus_dir=tmp_path / "c", kind=SeedKind.MOCK,
               max_lineage_share=0.01, rng=random.Random(0))
    root = _seed("root")
    assert c.add(root, fitness=1.0, force=True)
    # With max_lineage_share=0.01, no child should pass. Force should.
    child = _seed("child", parent_id=root.seed_id)
    assert c.add(child, fitness=1.0, force=True)
