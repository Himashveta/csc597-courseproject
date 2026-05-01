"""PyTorch seed: pointwise chain across an explicit dtype cast."""

import torch


def fn(x):
    y = torch.relu(x)
    y = torch.sigmoid(y)
    y = y.to(torch.float16)
    return y * 2


def main():
    x = torch.randn(128, 32)
    eager = fn(x)
    compiled = torch.compile(fn, backend="inductor")(x)
    if not torch.allclose(eager.float(), compiled.float(),
                          atol=1e-2, rtol=1e-2):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
