"""PyTorch seed: 2D conv with bias — exercises Inductor's conv dispatch."""

import torch
import torch.nn as nn


def main():
    m = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    x = torch.randn(1, 3, 16, 16)

    eager = m(x)
    compiled_m = torch.compile(m, backend="inductor")
    compiled = compiled_m(x)

    if not torch.allclose(eager, compiled, atol=1e-3, rtol=1e-3):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
