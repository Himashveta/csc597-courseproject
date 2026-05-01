"""Corpus = the collection of seeds the fuzzer has decided are worth keeping.

Adds proportional-to-fitness selection for the next mutation target.
A seed is added only if it produced novel coverage (driven externally
via UnifiedBitmap.update().is_novel()); the corpus itself does not
re-decide novelty.

We also enforce a soft diversity quota: no single (parent_id) lineage
may exceed `max_lineage_share` of the corpus. This addresses the
"alias_as_strided dominates" failure mode documented in the
dl-compiler-fuzzing skill — fertile seeds otherwise crowd out
everything else.
"""

from __future__ import annotations

import collections
import dataclasses
import pathlib
import random
from typing import Dict, List, Optional

from polyfuzz.corpus.seed import Seed, SeedKind


@dataclasses.dataclass
class CorpusEntry:
    seed: Seed
    fitness: float
    times_selected: int = 0


class Corpus:
    def __init__(
        self,
        corpus_dir: pathlib.Path,
        kind: SeedKind,
        max_lineage_share: float = 0.20,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._dir = corpus_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._kind = kind
        self._entries: Dict[str, CorpusEntry] = {}
        self._rng = rng or random.Random(0xC0FFEE)
        self._max_lineage_share = max_lineage_share

    # -- mutation ---------------------------------------------------

    def add(self, seed: Seed, fitness: float, force: bool = False) -> bool:
        """Add a seed if not already present. Writes it to disk.

        Returns True iff the seed was added.

        force=True bypasses the lineage diversity quota — used for
        initial seeds, which define the floor.
        """
        if seed.seed_id in self._entries:
            return False
        if not force and not self._lineage_ok(seed):
            return False
        seed.fitness = fitness
        seed.write(self._dir)
        self._entries[seed.seed_id] = CorpusEntry(seed=seed, fitness=fitness)
        return True

    # -- selection --------------------------------------------------

    def select(self) -> Optional[Seed]:
        """Pick the next seed to mutate, weighted by fitness.

        We apply two transformations to raw fitness to avoid the
        common failure mode where one early high-fitness seed monopo-
        lises mutation attention:

          1. Power-law compression: weight = (fitness + epsilon) ** 0.5
             — softens the gap between a seed that scored 250 on
             first sight and one that scored 3 on first sight.
          2. Times-selected decay: w' = w / (1 + 0.5 * times_selected)
             — gives less-explored seeds a real shot at being picked.

        Together these guarantee that every retained seed gets visited
        within a reasonable number of rounds.
        """
        if not self._entries:
            return None
        entries = list(self._entries.values())
        weights = [
            (max(0.01, e.fitness) + 0.5) ** 0.5
            / (1.0 + 0.5 * e.times_selected)
            for e in entries
        ]
        chosen = self._rng.choices(entries, weights=weights, k=1)[0]
        chosen.times_selected += 1
        return chosen.seed

    # -- queries ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def all_seeds(self) -> List[Seed]:
        return [e.seed for e in self._entries.values()]

    def stats(self) -> dict:
        if not self._entries:
            return {"size": 0}
        fitnesses = [e.fitness for e in self._entries.values()]
        return {
            "size": len(self._entries),
            "mean_fitness": sum(fitnesses) / len(fitnesses),
            "max_fitness": max(fitnesses),
            "min_fitness": min(fitnesses),
        }

    # -- internals --------------------------------------------------

    def _lineage_ok(self, seed: Seed) -> bool:
        """Reject if adding this seed would push its lineage past quota.

        The quota only takes effect once the corpus has enough entries
        for the share to be statistically meaningful — specifically,
        once `1/max_lineage_share` entries are present. With a 0.20
        cap, that's 5 entries before the quota engages. Below that
        threshold every child is accepted, since with a 2-entry corpus
        any single lineage trivially holds 100% of the corpus.
        """
        if seed.parent_id is None or not self._entries:
            return True
        if self._max_lineage_share <= 0:
            return True
        # Below this threshold the share is dominated by the discreteness
        # of corpus size, not by lineage diversity.
        min_corpus_for_quota = int(1.0 / self._max_lineage_share)
        if len(self._entries) < min_corpus_for_quota:
            return True

        counts: collections.Counter = collections.Counter()
        for e in self._entries.values():
            root = self._lineage_root(e.seed.seed_id)
            counts[root] += 1
        candidate_root = self._lineage_root(seed.parent_id)
        share = (counts.get(candidate_root, 0) + 1) / (len(self._entries) + 1)
        return share <= self._max_lineage_share

    def _lineage_root(self, sid: str) -> str:
        """Walk up the parent chain to find the root seed id."""
        while sid in self._entries:
            parent = self._entries[sid].seed.parent_id
            if parent is None:
                return sid
            sid = parent
        return sid
