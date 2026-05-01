"""PyTorch seed: sum reduction over a specified dim."""

import torch


def fn(x, dim):
    return x.sum(dim=dim)


def main():
    x = torch.randn(8, 16, 32)
    dim = 1
    eager = fn(x, dim)
    compiled = torch.compile(fn, backend="inductor")(x, dim)
    if not torch.allclose(eager, compiled, atol=1e-4, rtol=1e-3):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
