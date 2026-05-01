"""Seed = one self-contained Python program the fuzzer can execute.

Seeds are stored on disk as plain .py files. The Seed dataclass is the
in-memory handle: it carries the source code, an id, lineage, and a
fitness score after evaluation.

A seed file must be self-contained: when run with `python <file>` it
either exits 0 (no bug) or non-zero (potential bug). It must NOT
import polyfuzz itself — the fuzzer is the parent process, the seed
is the child.

Two kinds of seeds exist:
  - 'mock'   : drives the mock_compiler target
  - 'pytorch': drives torch.compile

The kind is informational; the runner doesn't dispatch on it.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import pathlib
from typing import Optional, Tuple


class SeedKind(str, enum.Enum):
    MOCK = "mock"
    PYTORCH = "pytorch"


@dataclasses.dataclass
class Seed:
    seed_id: str
    kind: SeedKind
    source: str
    parent_id: Optional[str] = None
    mutator_used: Optional[str] = None
    generation: int = 0
    fitness: float = 0.0

    @classmethod
    def from_source(
        cls,
        source: str,
        kind: SeedKind,
        parent_id: Optional[str] = None,
        mutator_used: Optional[str] = None,
        generation: int = 0,
    ) -> "Seed":
        digest = hashlib.blake2b(source.encode(), digest_size=8).hexdigest()
        return cls(
            seed_id=digest,
            kind=kind,
            source=source,
            parent_id=parent_id,
            mutator_used=mutator_used,
            generation=generation,
        )

    def write(self, dirpath: pathlib.Path) -> pathlib.Path:
        """Write the seed source to dirpath/<id>.py and return the path."""
        dirpath.mkdir(parents=True, exist_ok=True)
        out = dirpath / f"{self.seed_id}.py"
        out.write_text(self.source)
        return out

    @classmethod
    def load(cls, path: pathlib.Path, kind: SeedKind) -> "Seed":
        src = path.read_text()
        return cls.from_source(src, kind=kind)

    def header_lines(self) -> Tuple[str, ...]:
        return (
            f"# polyfuzz seed_id={self.seed_id}",
            f"# kind={self.kind.value}",
            f"# generation={self.generation}",
            f"# parent={self.parent_id or '-'}",
            f"# mutator={self.mutator_used or '-'}",
        )
