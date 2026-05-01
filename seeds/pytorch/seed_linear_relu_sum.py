"""PyTorch seed: tiny torch.compile workload.

Seeds for the PYTORCH_TARGET drive torch.compile through Dynamo +
Inductor. Each seed must:
  - exit 0 on success
  - exit non-zero (or crash) on failure

The fuzzer mutates the literal numbers, dtypes, and op types. To extend
the seed library, drop more files of this shape into seeds/pytorch/.

Note: this seed is just a template — running it under PolyFuzz only
yields useful coverage if PyTorch was built with --coverage.
"""

import torch
import torch.nn as nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(32, 16)

    def forward(self, x):
        return torch.relu(self.lin(x)).sum(dim=1)


def main():
    m = Tiny()
    x = torch.randn(4, 32)

    eager = m(x)
    compiled_fn = torch.compile(m, backend="inductor")
    compiled = compiled_fn(x)

    if not torch.allclose(eager, compiled, atol=1e-4, rtol=1e-3):
        # Differential mismatch counts as a bug.
        raise SystemExit(2)


if __name__ == "__main__":
    main()
