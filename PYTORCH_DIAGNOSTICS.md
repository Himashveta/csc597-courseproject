# PyTorch Diagnostics Notes

This artifact includes `scripts/run_pytorch.py`, but the current reliable end-to-end evaluation is the mock target, not real PyTorch.

## What succeeded on the server

The PyTorch preflight detected a coverage-instrumented source build:

```text
OK gcov_source_root: ok: 6505 .gcno files under /scratch/hkk5340/pytorch-ptcov
OK py_packages: ok: importable: ['torch._dynamo', 'torch._inductor', 'torch.fx']
```

Direct eager-backend seeds also ran successfully after changing the PyTorch seeds from `backend="inductor"` to `backend="eager"`.

## What failed

Running those same seeds through `coverage.py` crashed during `import torch`, before seed logic executed:

```text
File "/scratch/hkk5340/pytorch-ptcov/torch/__init__.py", line 444, in <module>
  from torch._C import *
SystemError: ... method: bad call flags
Segmentation fault
```

Therefore `merged_bitmap` cannot collect reliable PyTorch Python coverage on that build/server combination.

## Branch-state status

Full PyTorch `branch_state` is blocked because PyTorch lacks C-side `MC_PROBE` branch-variable instrumentation. The supported diagnostic fallback is:

```bash
python3 scripts/run_pytorch.py \
  --feedback-mode branch_state \
  --branch-state-c-disabled \
  --smoke-only
```

This gives Python-only branch-state and should be described as a degraded signal, not full PolyFuzz-style cross-language branch-state.
