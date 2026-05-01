# PolyFuzz for Deep-Learning Compilers

This repository contains a course/research artifact that adapts ideas from **PolyFuzz: Holistic Greybox Fuzzing of Multi-Language Systems** to the deep-learning compiler setting, where a single `torch.compile` workload crosses both Python compiler frontend code and native C/C++ backend code.

The artifact compares two iterations of the same fuzzer through a shared `feedback_mode` interface:

1. **`merged_bitmap`** — a strawman AFL-style feedback mode that merges Python coverage from `coverage.py` and C/C++ coverage from `gcov` into one 64 KiB bitmap.
2. **`branch_state`** — a PolyFuzz-inspired branch-state mode that tracks `(branch_id, value_class_hash)` events. Python branch variables are harvested with `sys.settrace`; C branch variables are harvested with `MC_PROBE` macros in the mock target.

The main supported end-to-end evaluation is on a small mock DL compiler target. The PyTorch entry point is included as a diagnostic/obstacle reporter rather than a source of final PyTorch numbers.

---

## Repository layout

```text
.
├── src/polyfuzz/              # Main Python package
│   ├── fuzzer.py              # PolyFuzz class and feedback_mode switch
│   ├── target.py              # MOCK_TARGET and PYTORCH_TARGET specs
│   ├── coverage/              # Python/C coverage + branch-state collectors
│   ├── corpus/                # Seed/corpus management
│   ├── mutators/              # AST/text-level seed mutators
│   ├── oracle/                # Crash/sanitizer/assertion classifier
│   └── harness/               # Subprocess runner and seed bootstrap
├── target/                    # Mock C compiler + Python frontend
├── seeds/mock/                # Mock-target seed programs
├── seeds/pytorch/             # PyTorch template seeds
├── scripts/
│   ├── run_mock.py            # Single mock-target fuzz run
│   ├── multi_trial.py         # Four-way ablation driver
│   └── run_pytorch.py         # PyTorch preflight/obstacle reporter
├── tests/                     # Unit tests
├── report/polyfuzz.md         # Project writeup
├── Makefile                   # Reproduction entry point
├── Dockerfile                 # Containerized mock-target workflow
└── requirements.txt
```

---

## Requirements

For the mock target:

- Python 3.10+
- `gcc`, `make`, `gcov`
- `pytest`
- `coverage.py`

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

If your system only has `python3` and not `python`, pass `PYTHON=python3` to Make targets.

---

## Quick start: mock target

```bash
make PYTHON=python3 build
make PYTHON=python3 install
make PYTHON=python3 test
make PYTHON=python3 demo
```

Or run all of the above:

```bash
make PYTHON=python3 build install test demo
```

`make demo` runs a short branch-state fuzzing session against the mock compiler and writes output under `results/demo/`.

---

## Reproduce the four-way ablation

```bash
make PYTHON=python3 eval TRIALS=3 BUDGET_SEC=22
```

This runs four variants:

| Variant | Mode | Signal |
|---|---|---|
| `cc_only` | `merged_bitmap` | C/C++ coverage only |
| `py_only` | `merged_bitmap` | Python coverage only |
| `merged_bitmap` | `merged_bitmap` | Python + C/C++ coverage merged into one bitmap |
| `branch_state` | `branch_state` | Python + C branch-variable/value-class events |

Expected output files:

```text
results/multi_trial/aggregate.json
results/multi_trial/trial_*_*/polyfuzz.log.jsonl
results/multi_trial/trial_*_*/bugs/
results/multi_trial/trial_*_*/corpus/
```

The project writeup reports the headline result that `branch_state` retains substantially more seeds than `merged_bitmap` on the mock target while finding comparable unique bug classes.

---

## PyTorch mode: what works and what does not

The PyTorch script is intentionally conservative. It checks whether prerequisites are present and refuses to publish meaningless zero-coverage results when they are not.

Run the preflight/smoke check:

```bash
export TORCH_ROOT=/path/to/pytorch-coverage-build
python3 scripts/run_pytorch.py --feedback-mode merged_bitmap --smoke-only
```

For branch-state without C probes:

```bash
python3 scripts/run_pytorch.py \
  --feedback-mode branch_state \
  --branch-state-c-disabled \
  --smoke-only
```

### Important PyTorch limitations

- A standard `pip install torch` is not enough for C/C++ coverage because it does not include `.gcno` files.
- `merged_bitmap` needs a PyTorch build compiled with gcov coverage flags.
- Full `branch_state` needs C-side branch-variable probes. The mock target has `MC_PROBE` macros; PyTorch does not. Therefore full PyTorch branch-state is blocked unless PyTorch is instrumented with an equivalent source patch or LLVM pass.
- On our server, direct eager-backend PyTorch seeds ran successfully, but `coverage.py` crashed while importing the coverage-built `torch._C` extension. In that environment, PyTorch execution is useful as a diagnostic, not as a final quantitative evaluation.

The obstacle report is written to:

```text
results/pytorch/obstacle_report.json
```

---

## Building a gcov PyTorch for diagnostics

A minimal CPU-only direction is:

```bash
git clone --depth 1 https://github.com/pytorch/pytorch.git ~/pytorch-ptcov
cd ~/pytorch-ptcov
git submodule update --init --recursive --depth 1

python3 -m venv ~/ptcov-venv
source ~/ptcov-venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install cmake ninja
python3 -m pip install -r requirements.txt

export USE_CUDA=0
export USE_DISTRIBUTED=0
export USE_NCCL=0
export USE_MKLDNN=1
export USE_FBGEMM=0
export BUILD_TEST=0
export USE_KINETO=0
export USE_GCOV=1
export CFLAGS="--coverage -O0 -g"
export CXXFLAGS="--coverage -O0 -g"
export LDFLAGS="--coverage"
export DEBUG=1
export MAX_JOBS=4

python3 -m pip install --no-build-isolation -e .
```

Verify:

```bash
export TORCH_ROOT=$HOME/pytorch-ptcov
python3 -c "import torch; print(torch.__version__); print(torch.__file__)"
find "$TORCH_ROOT" -name '*.gcno' | head
```

---

## Docker workflow

```bash
docker build -t polyfuzz-dlcompilers .
docker run --rm polyfuzz-dlcompilers make PYTHON=python3 test
docker run --rm polyfuzz-dlcompilers make PYTHON=python3 eval TRIALS=3 BUDGET_SEC=22
```

---

## Claims and reproduction map

| Claim | Command/output |
|---|---|
| Unit tests pass | `make PYTHON=python3 test` |
| Mock target builds with gcov + UBSan + probes | `make PYTHON=python3 build` |
| Four-way ablation runs | `make PYTHON=python3 eval TRIALS=3 BUDGET_SEC=22` |
| Branch-state retains more seeds | `results/multi_trial/aggregate.json` |
| Mock divide-by-zero bug is found | `results/multi_trial/trial_*_*/bugs/` |
| PyTorch prerequisites are diagnosed | `python3 scripts/run_pytorch.py --feedback-mode merged_bitmap --smoke-only` |
| Full PyTorch branch-state is blocked by missing C probes | `python3 scripts/run_pytorch.py --feedback-mode branch_state --smoke-only` |

---

## Notes for GitHub upload

Before committing, avoid adding generated outputs:

```bash
rm -rf results/ build/ dist/ .pytest_cache/ **/__pycache__
```

Then:

```bash
git init
git add .
git commit -m "Add PolyFuzz DL compiler artifact"
```

---

## License

This artifact is released under the MIT License. See [`LICENSE`](LICENSE).
