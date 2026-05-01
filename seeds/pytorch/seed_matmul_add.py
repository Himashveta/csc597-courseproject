"""PyTorch seed: matmul + add fused via Inductor."""

import torch


def fn(x, y, b):
    return torch.matmul(x, y) + b


def main():
    x = torch.randn(64, 32)
    y = torch.randn(32, 16)
    b = torch.randn(64, 16)

    eager = fn(x, y, b)
    compiled = torch.compile(fn, backend="inductor")(x, y, b)

    if not torch.allclose(eager, compiled, atol=1e-4, rtol=1e-3):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
