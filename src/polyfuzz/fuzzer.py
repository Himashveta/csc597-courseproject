"""PolyFuzz — the main fuzzer class.

Two iterations of the same idea live behind one class, selectable
via the `feedback_mode` constructor argument:

  - 'merged_bitmap' (Iteration 1): AFL-style bitmap fed by line and
     edge coverage from coverage.py + gcov. The fitness signal is
     "did this seed set new bits in the merged bitmap?". This is the
     baseline we compare PolyFuzz against.

  - 'branch_state' (Iteration 2): PolyFuzz-style branch-variable
     feedback. Python branches are harvested via sys.settrace; C
     branches are harvested via explicit MC_PROBE macros. Each
     observation is bucketed into a value class and aggregated into
     a set of (branch_id, value_class) pairs. Novelty is set growth.

Architecture (left to right):

    initial_seeds  ->  Corpus  ->  select & mutate  ->  Seed.write  ->
    SeedRunner.run (subprocess + GCOV_PREFIX + MC_TRACE_FD)         ->
    PythonCoverage / CppCoverage / branch_vars / c_branch_vars      ->
    UnifiedBitmap.update OR BranchStateSet.update                   ->
    Oracle.classify_outcome                                         ->
    fitness = (mode-dependent linear combo) + bug_bonus             ->
    Corpus.add (only if novel, or force_keep for initial seeds)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import random
import time
from typing import List, Optional

from polyfuzz.corpus.corpus import Corpus
from polyfuzz.corpus.seed import Seed, SeedKind
from polyfuzz.coverage.branch_state import BranchStateDelta, BranchStateSet
from polyfuzz.coverage.branch_vars import read_py_events
from polyfuzz.coverage.c_branch_vars import parse_trace_file
from polyfuzz.coverage.cpp_cov import CppCoverage
from polyfuzz.coverage.python_cov import PythonCoverage
from polyfuzz.coverage.unified import CoverageDelta, UnifiedBitmap
from polyfuzz.harness.runner import SeedRunner, SeedRunOutcome
from polyfuzz.mutators.mutators import Mutator, get_mutator_registry
from polyfuzz.oracle.oracle import OracleVerdict, classify_outcome
from polyfuzz.target import TargetSpec


log = logging.getLogger("polyfuzz")


@dataclasses.dataclass
class FitnessWeights:
    """Linear weights for the fitness function.

    Iteration 1 ('merged_bitmap') uses w_py_line, w_py_arc, w_cc_line,
    w_cc_branch. Iteration 2 ('branch_state') uses w_py_event and
    w_c_event. The bug bonus is shared.
    """
    # Iteration 1 (merged_bitmap)
    w_py_line: float = 1.0
    w_py_arc: float = 0.5
    w_cc_line: float = 1.0
    w_cc_branch: float = 0.5

    # Iteration 2 (branch_state)
    w_py_event: float = 1.0
    w_c_event: float = 1.0

    # Shared
    w_bug_bonus: float = 10.0


@dataclasses.dataclass
class FuzzStats:
    iterations: int = 0
    seeds_added: int = 0
    bugs_found: int = 0
    timeouts: int = 0
    last_new_iteration: int = 0
    started_at: float = 0.0


class PolyFuzz:
    """Coverage- or branch-state-guided fuzzer."""

    VALID_MODES = ("merged_bitmap", "branch_state")

    def __init__(
        self,
        target: TargetSpec,
        output_dir: pathlib.Path,
        kind: SeedKind = SeedKind.MOCK,
        feedback_mode: str = "branch_state",
        bitmap_size: int = UnifiedBitmap.DEFAULT_SIZE,
        fitness_weights: Optional[FitnessWeights] = None,
        seed: int = 0xC0FFEE,
        mutators: Optional[List[Mutator]] = None,
        skip_target_check: bool = False,
    ) -> None:
        if feedback_mode not in self.VALID_MODES:
            raise ValueError(
                f"feedback_mode must be one of {self.VALID_MODES}, "
                f"got {feedback_mode!r}"
            )
        self._feedback_mode = feedback_mode
        self._target = target
        if not skip_target_check:
            problem = target.sanity_check()
            if problem is not None:
                raise FileNotFoundError(
                    f"Target {target.name} not usable: {problem}. "
                    f"Pass skip_target_check=True to bypass."
                )
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._runs_dir = output_dir / "runs"
        self._bug_dir = output_dir / "bugs"
        self._bug_dir.mkdir(parents=True, exist_ok=True)
        self._corpus_dir = output_dir / "corpus"
        self._log_path = output_dir / "polyfuzz.log.jsonl"

        self._kind = kind
        self._weights = fitness_weights or FitnessWeights()
        self._rng = random.Random(seed)

        self._py_cov = PythonCoverage(target.py_source_packages)
        self._cc_cov = CppCoverage(
            gcov_source_root=target.gcov_source_root,
            include_prefixes=target.gcov_include_prefixes,
            exclude_prefixes=target.gcov_exclude_prefixes,
        )
        self._bitmap = UnifiedBitmap(size=bitmap_size)
        self._branch_state = BranchStateSet()

        self._corpus = Corpus(self._corpus_dir, kind=kind, rng=self._rng)
        runner_mode = (
            "merged_bitmap" if feedback_mode == "merged_bitmap"
            else "branch_state"
        )
        self._runner = SeedRunner(
            target=target, runs_dir=self._runs_dir,
            feedback_mode=runner_mode,
        )
        self._mutators = mutators or get_mutator_registry()

        self._stats = FuzzStats(started_at=time.time())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed_initial(self, source_dir: pathlib.Path) -> int:
        if not source_dir.exists():
            raise FileNotFoundError(f"seed dir {source_dir} not found")
        files = sorted(source_dir.glob("seed_*.py"))
        if not files:
            raise ValueError(f"no seed_*.py files found under {source_dir}")
        retained = 0
        for path in files:
            src = path.read_text()
            seed = Seed.from_source(src, kind=self._kind, generation=0)
            if self._evaluate_and_consider(seed, force_keep=True):
                retained += 1
        log.info("seeded corpus: %d/%d retained", retained, len(files))
        return retained

    def run(
        self,
        time_budget_sec: float = 60.0,
        max_iterations: Optional[int] = None,
    ) -> FuzzStats:
        if len(self._corpus) == 0:
            raise RuntimeError("corpus empty — call seed_initial() first.")
        deadline = time.time() + time_budget_sec
        while time.time() < deadline:
            if max_iterations is not None \
                    and self._stats.iterations >= max_iterations:
                break
            parent = self._corpus.select()
            if parent is None:
                break
            mutator, child = self._mutate(parent)
            if child is None:
                continue
            self._evaluate_and_consider(child)
        self._write_summary()
        return self._stats

    def stats_dict(self) -> dict:
        cov_summary = (
            self._bitmap.coverage_summary() if self._feedback_mode == "merged_bitmap"
            else self._branch_state.coverage_summary()
        )
        return {
            "iterations":     self._stats.iterations,
            "seeds_added":    self._stats.seeds_added,
            "bugs_found":     self._stats.bugs_found,
            "timeouts":       self._stats.timeouts,
            "corpus":         self._corpus.stats(),
            "feedback_mode":  self._feedback_mode,
            "feedback":       cov_summary,
            "elapsed_sec":    time.time() - self._stats.started_at,
            "target":         self._target.name,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _mutate(self, parent: Seed):
        weights = [m.weight for m in self._mutators]
        for _ in range(8):
            mutator = self._rng.choices(self._mutators, weights=weights, k=1)[0]
            new_src = mutator.apply(parent.source, self._rng)
            if new_src is None or new_src == parent.source:
                continue
            child = Seed.from_source(
                new_src,
                kind=parent.kind,
                parent_id=parent.seed_id,
                mutator_used=mutator.name,
                generation=parent.generation + 1,
            )
            return mutator, child
        return None, None

    def _evaluate_and_consider(
        self,
        seed: Seed,
        force_keep: bool = False,
    ) -> bool:
        seed_path = seed.write(self._corpus_dir / "_pending")
        outcome = self._runner.run(seed.seed_id, seed_path)
        self._stats.iterations += 1
        if outcome.timed_out:
            self._stats.timeouts += 1

        if self._feedback_mode == "merged_bitmap":
            delta = self._update_merged_bitmap(outcome)
        else:
            delta = self._update_branch_state(outcome)

        verdict = classify_outcome(
            outcome.return_code, outcome.stderr_tail, outcome.timed_out,
        )
        is_bug = verdict.is_bug()
        fitness = self._fitness(delta, is_bug)

        kept = False
        if delta.is_novel() or force_keep:
            kept = self._corpus.add(seed, fitness=fitness, force=force_keep)
            if kept:
                self._stats.seeds_added += 1
                self._stats.last_new_iteration = self._stats.iterations

        if is_bug:
            self._record_bug(seed, outcome, verdict, delta)
            self._stats.bugs_found += 1

        self._log_event(seed, outcome, verdict, delta, fitness, kept)
        return kept

    def _update_merged_bitmap(self, outcome: SeedRunOutcome) -> CoverageDelta:
        py = self._py_cov.read(outcome.coverage_data_file)
        cc = self._cc_cov.read(outcome.gcov_prefix_dir)
        return self._bitmap.update(py, cc)

    def _update_branch_state(self, outcome: SeedRunOutcome) -> BranchStateDelta:
        py = read_py_events(outcome.py_events_file)
        cc = parse_trace_file(outcome.c_trace_file)
        return self._branch_state.update(py, cc)

    def _fitness(self, delta, is_bug: bool) -> float:
        w = self._weights
        if isinstance(delta, CoverageDelta):
            base = (
                w.w_py_line * delta.new_py_lines
                + w.w_py_arc * delta.new_py_arcs
                + w.w_cc_line * delta.new_cc_lines
                + w.w_cc_branch * delta.new_cc_branches
            )
        else:
            base = (
                w.w_py_event * delta.new_py_events
                + w.w_c_event * delta.new_c_events
            )
        if is_bug:
            base += w.w_bug_bonus
        return base

    def _record_bug(
        self,
        seed: Seed,
        outcome: SeedRunOutcome,
        verdict: OracleVerdict,
        delta,
    ) -> None:
        bug_id = f"{verdict.bug_class.value}_{seed.seed_id}"
        bdir = self._bug_dir / bug_id
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "seed.py").write_text(seed.source)
        (bdir / "report.json").write_text(json.dumps({
            "seed_id":      seed.seed_id,
            "parent_id":    seed.parent_id,
            "mutator":      seed.mutator_used,
            "generation":   seed.generation,
            "bug_class":    verdict.bug_class.value,
            "summary":      verdict.summary,
            "return_code":  verdict.return_code,
            "signal":       verdict.signal_name,
            "duration_sec": outcome.duration_sec,
            "delta":        self._delta_to_dict(delta),
            "stderr_tail":  outcome.stderr_tail[-1500:],
        }, indent=2))

    def _log_event(
        self,
        seed: Seed,
        outcome: SeedRunOutcome,
        verdict: OracleVerdict,
        delta,
        fitness: float,
        kept: bool,
    ) -> None:
        rec = {
            "iter":            self._stats.iterations,
            "seed_id":         seed.seed_id,
            "parent_id":       seed.parent_id,
            "mutator":         seed.mutator_used,
            "generation":      seed.generation,
            "bug_class":       verdict.bug_class.value,
            "return_code":     verdict.return_code,
            "duration_sec":    round(outcome.duration_sec, 4),
            "feedback_mode":   self._feedback_mode,
            "fitness":         round(fitness, 3),
            "kept":            kept,
            **self._delta_to_dict(delta),
        }
        with self._log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    @staticmethod
    def _delta_to_dict(delta) -> dict:
        if isinstance(delta, CoverageDelta):
            return {
                "new_py_lines":    delta.new_py_lines,
                "new_py_arcs":     delta.new_py_arcs,
                "new_cc_lines":    delta.new_cc_lines,
                "new_cc_branches": delta.new_cc_branches,
                "new_bits":        delta.new_bitmap_bits,
            }
        return {
            "new_py_events": delta.new_py_events,
            "new_c_events":  delta.new_c_events,
        }

    def _write_summary(self) -> None:
        path = self._output_dir / "summary.json"
        path.write_text(json.dumps(self.stats_dict(), indent=2))


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
