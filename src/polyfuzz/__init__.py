"""PolyFuzz — coverage-guided fuzzing across the Python/C language boundary.

Public surface:
    from polyfuzz import PolyFuzz, TargetSpec, Seed, SeedKind
"""

from polyfuzz.fuzzer import PolyFuzz
from polyfuzz.target import TargetSpec, MOCK_TARGET, PYTORCH_TARGET
from polyfuzz.corpus.seed import Seed, SeedKind

__all__ = ["PolyFuzz", "TargetSpec", "Seed", "SeedKind",
           "MOCK_TARGET", "PYTORCH_TARGET"]
__version__ = "0.1.0"
