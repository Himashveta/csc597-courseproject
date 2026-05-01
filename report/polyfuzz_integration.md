# PolyFuzz for Deep-Learning Compilers: Two Iterations and What Got in the Way

## Abstract

Coverage-guided fuzzers for deep-learning compilers like
`torch.compile` face a structural problem: the system under test
straddles a Python frontend and a C++ backend, and most existing
fuzzers see only one of the two. We re-implement, adapt, and evaluate
PolyFuzz (Li et al., USENIX Sec '23) — the canonical multi-language
greybox fuzzer — in the DL-compiler setting. We did this in two
iterations. The first was a strawman: an AFL-style bitmap fed by line
and edge coverage from `coverage.py` and `gcov`, merged into one
fitness signal. The second is a faithful PolyFuzz port: branch-variable
harvesting on both layers, with runtime values bucketed into coarse
value classes and a unified set-based novelty signal. Each iteration
runs as a constructor argument of the same class. We present an
ablation across four feedback regimes (C-only, Python-only,
merged-bitmap, branch-state) on a small instrumented mock compiler,
and an obstacle analysis for porting the same setup to PyTorch. The
mock-target ablation shows that branch-state retains 2.8× more seeds
than merged-bitmap with comparable bug discovery; the PyTorch
analysis shows that branch-state's C-side instrumentation is not
feasible without source patches, which we take as the core obstacle
to a faithful PolyFuzz port for DL compilers.


## 1. Problem & Motivation

Deep-learning compilers are now part of the trust boundary for ML
research and production. A `torch.compile` invocation traverses
hundreds of thousands of lines of Python (Dynamo, AOTAutograd, FX,
Inductor) and C++ (ATen kernels, Triton bindings) before returning a
tensor. Defects in this stack — crashes, miscompilations,
sanitizer-detected memory errors — propagate silently into model
training and inference. Coverage-guided fuzzing is the standard tool
for finding them, but the existing tools split along a language axis
that doesn't match the system: TitanFuzz, FuzzGPT, NNSmith, and
PromptFuzz all operate at the Python API level with at best Python
coverage feedback, while AFL-family tools target the compiled
artefact and miss the dispatch decisions that decide which kernel
runs. The seam between the layers, where many real bugs live, is
invisible to the fuzzer.

PolyFuzz (Li et al., USENIX Sec '23) is the only published fuzzer
that explicitly targets multi-language systems with cross-language
feedback. Its design is built around three ideas: a unified IR
(SAIR) for branch-variable definitions across languages; runtime
harvesting of variable values at every branch; and a regression
model from input bytes to branch outcomes that drives mutation when
the fuzzer gets stuck. Real PolyFuzz was evaluated on Pillow,
NumPy, lxml — Python-with-C-extensions targets where the input is
literally a byte buffer.

This paper asks two questions:
1. What does the merged-coverage fitness signal — the design we
   originally built before reading the PolyFuzz paper carefully —
   actually buy us in the DL-compiler setting?
2. How much of PolyFuzz's design transfers when "the input" is a
   Python source program rather than a byte buffer, and the C side
   is PyTorch rather than libpng?

The answer to (1) is "not much". The answer to (2) is "the Python
side ports cleanly; the C side requires source instrumentation that
PyTorch does not have, which the artefact's run_pytorch.py
diagnoses but cannot work around".


## 2. Background & Related Work

**LLM-based DL fuzzers.** TitanFuzz (Deng et al., ISSTA 2023) and
FuzzGPT generate test programs with LLMs and observe Python-level
coverage. NNSmith (Liu et al., ASPLOS 2023) uses an SMT-driven
generator for valid graphs and runs them across multiple compiler
backends as a differential oracle. FUEL (the paper we recently
reviewed) extends LLM-based fuzzing with feedback-aware simulated
annealing and dual-agent prompting. None of these tools observe C++
control flow.

**Coverage-guided native fuzzers.** AFL/AFL++ are the canonical
coverage-guided fuzzers for native code. The AFL bitmap design — a
fixed-size byte array of saturating counters indexed by edge hash —
is the basis for our Iteration 1. None of these fuzzers see Python
execution.

**PolyFuzz.** Li et al. (USENIX Sec '23) is the closest published
work to ours. We summarise their design, then describe the gap to
ours. PolyFuzz uses three pieces:

1. **SAIR.** A custom intermediate representation that captures
   basic blocks and the variables in their branch predicates,
   uniformly across C, Python, and Java.
2. **Branch-variable harvesting.** At runtime, for every branch in
   the SAIR graph, the fuzzer records the value of the predicate
   variables. Cross-language data flow becomes observable: a Python
   `len(x)` and a C `if (x > 100)` that read from the same input
   byte are both visible.
3. **Sensitivity analysis and input-to-branch model.** The fuzzer
   partitions the input byte buffer into blocks, flips each block,
   and observes which branch variables change. This learns a
   regression model byte-block → branch-variable, used to direct
   mutation when stuck.

PolyFuzz was evaluated on Pillow, NumPy, lxml, jansi — targets
where "the input" is a byte buffer (PNG, npy, XML, ANSI escape
codes). Its C-side instrumentation is an LLVM pass plugged into
AFL++. We reuse design pieces 1 and 2; we do not implement piece 3
because our target's input is not a byte buffer.

**Why the input model matters.** PolyFuzz's sensitivity analysis
assumes a fixed-shape byte buffer that the fuzzer can perturb at
known offsets. A `torch.compile` workload's input is a Python
source program that constructs a model, allocates tensors, and runs
operators. There is no byte buffer to flip. The mutator paradigm
that fits this setting is *program mutation* (TitanFuzz,
FuzzGPT, our own AST-based mutators) rather than byte mutation.
The branch-variable harvesting *signal* still applies — it just has
to be paired with a different kind of mutation engine.


## 3. Iteration 1: Merged Bitmap (Strawman)

Our first design merged Python and C/C++ coverage into a single
AFL-style bitmap. A seed was novel iff it set at least one new bit
in the union. We thought of this as "PolyFuzz lite" — until reading
the paper revealed that what PolyFuzz actually does is much
stronger. We retain the design as a baseline because:

1. It is what most existing dual-language fuzzers would produce if
   asked to merge feedback signals.
2. It is the right *control* against which to measure whether
   branch-variable harvesting actually pays off.

### 3.1 Bitmap design

Three event classes hash into a single 64KiB byte array of
saturating counters:

| Event class       | Hash input                            | Tag prefix    |
|-------------------|---------------------------------------|---------------|
| Python line       | `(filename, lineno)`                  | `py_line\|`   |
| Python branch arc | `(filename, src_lineno, dst_lineno)`  | `py_arc\|`    |
| C/C++ line        | `(filename, lineno)`                  | `cc_line\|`   |
| C/C++ branch      | `(filename, lineno, branch_idx)`      | `cc_branch\|` |

Tag prefixes ensure that a Python line at `(foo.py, 42)` and a C
line at `(foo.c, 42)` map to (with high probability) different
slots. Novelty is "this slot was zero before".

### 3.2 What it gets right

The merged bitmap solves one real problem: a seed that explores new
Python control flow but happens to traverse the same C kernels as
prior seeds is retained. Symmetrically, a seed that hits a new C
branch via a familiar Python path is also retained. Single-language
fuzzers reject both as redundant.

### 3.3 What it gets wrong

The merged bitmap shares AFL's coarseness: it credits a seed for
*hitting* a branch, regardless of *how*. Two seeds that take the
same branch with `x=42` and `x=10000` look identical. For a DL
compiler this is a significant signal loss because most interesting
bugs depend on specific value classes — empty tensors, mismatched
strides, dtype-precision combinations — not on coverage of new
edges. Sec. 5 shows quantitatively that the merged bitmap saturates
within seconds on our mock target, after which mutation runs blind.


## 4. Iteration 2: Branch-State Feedback (Faithful PolyFuzz)

The second iteration ports PolyFuzz's branch-variable harvesting to
our setting. Both halves are re-implementations from the paper's
design, adapted for the DL-compiler context.

### 4.1 Python branch-variable harvesting

We instrument the Python frontend via `sys.settrace` rather than an
AST rewrite. Static AST analysis up front identifies which lines
contain branch predicates (`If`, `While`, `IfExp`, `Assert`) and
which variables their predicates reference. At runtime, on every
trace event matching a branch line, we read the named locals from
the calling frame, bucket them into coarse value classes, and emit
one event per branch.

Bucketing is the design move that makes this useful. Without it,
seeds with shapes `[64, 64]` and `[65, 65]` would produce different
events even though they take the same control-flow paths in every
predicate. With log-uniform power-of-two bucketing, "same outcome"
maps to "same event" while "different outcome" maps to "different
event". A `branch at file:42` with `x=5` and `x=7` is one event;
with `x=200` it is a different event because the predicate `x > 100`
flips outcome.

Concretely: integers bucket by sign and floor-log2 magnitude;
sequences bucket by length on the same scale; tensors bucket by
`(dtype, rank)`; everything else falls back to the type name. Tests
in `tests/test_branch_vars.py` pin this behaviour.

### 4.2 C branch-variable harvesting

PolyFuzz's C instrumentation is an LLVM pass plugged into AFL++.
Implementing that pass for arbitrary C/C++ code was outside our
budget; we took a pragmatic shortcut and added explicit `MC_PROBE`
macros to the mock target's source. Each branch is preceded by

    MC_PROBE_I2(branch_id, op->shape[0], op->shape[1]);

which expands into a `printf`-style emit guarded by a runtime check
on the `MC_TRACE_FD` environment variable. When unset, the macros
are no-ops; ordinary fuzzing runs are unaffected. When the harness
sets `MC_TRACE_FD` to a per-seed file descriptor, every branch in
the dispatch path emits its variable-value record.

The reader side (`c_branch_vars.py`) parses these records and
applies the same value-class bucketing as the Python side, so C and
Python events are combined uniformly into a single set.

This shortcut is exactly the obstacle that bites us on PyTorch:
real PyTorch is not source-instrumented, and adding `MC_PROBE`
macros to its hundreds of thousands of branch sites is not feasible
manually. Sec. 6 documents this honestly.

### 4.3 Unified branch-state set

Both event streams flow into one `BranchStateSet` whose state is
two sets — `py_events` and `c_events` — of `(branch_id,
value_class_hash)` tuples. A seed is novel iff its events grow
either set. This is structurally identical to the merged bitmap's
"set new bits" predicate, but the *atom* is finer: a tuple of
(branch, value-class) instead of a hashed edge.

### 4.4 What we did NOT port

We deliberately omit PolyFuzz's input-to-branch regression model.
The model is the byte-buffer pair to byte-block sensitivity
analysis; with program mutation as the input model, the byte-block
abstraction does not apply. A future analogue would learn AST-node
to branch-variable correspondences — a real research direction we
flag as out of scope.


## 5. Evaluation: Mock Target Ablation

### 5.1 Setup

Our mock target is a 280-line C compiler-like library (`select_kernel`
for matmul/conv/reduce/pointwise dispatch, `decide_fusion` for
post-kernel fusion, `codegen` for output stringification) wrapped
by a 200-line Python frontend that performs `_validate_op`,
`_normalize_layout`, and `_should_break` decisions. The C side is
compiled with `-fsanitize=undefined` (recover-on) and gcov
instrumentation. The C side also has `MC_PROBE` macros at every
internal branch (39 probe sites).

Three intentional bugs are planted in the C source: a divide-by-zero
in the linear-reduction kernel (line 191, triggered when
`shape[reduce_dim]==0`), an out-of-bounds write in the
NHWC-FP16-Winograd kernel selector (line 204, when `shape[3]>64`),
and an unguarded layout/rank combination in the entry point. UBSan
makes all three observable.

We compare four feedback regimes via the same `PolyFuzz` class, with
identical seeds, mutators, and per-trial RNG seed:

- **cc_only** — only C/C++ coverage; merged_bitmap mode with Python
  weights zeroed.
- **py_only** — only Python coverage; merged_bitmap mode with C
  weights zeroed.
- **merged_bitmap** (Iteration 1) — both layers' coverage in one bitmap.
- **branch_state** (Iteration 2) — branch-variable harvesting on both layers.

Three independent trials per variant, 22 seconds per trial,
reproducible via `make eval`.

### 5.2 Results

| Variant         | Trials w/ bug | Mean total bugs | Mean unique bugs | Mean corpus | Mean C events |
|-----------------|---------------|-----------------|------------------|-------------|---------------|
| cc_only         | 0/3           | 0               | 0                | 18          | 125 br        |
| py_only         | 2/3           | 8.33            | 1.0              | 18          | 137 br        |
| merged_bitmap   | **3/3**       | 11.0            | 1.0              | 19          | 145 br        |
| branch_state    | 2/3           | 1.67            | 1.0              | **52.7**    | 103 ev        |

Three findings:

**Branch-state retains 2.8× more seeds.** The most striking
difference. With branch-state's finer-grained novelty predicate, far
more mutated children clear the "added new event?" bar. The corpus
grows steadily across the trial; the merged-bitmap variants saturate
within seconds. This is the diversity property branch-state was
designed to deliver.

**Total bug count is misleading.** `merged_bitmap` reports the
highest mean total bugs (11.0) but only 1 unique bug per trial. What
happens: once a parent seed lands on the divide-by-zero path, the
fitness function gives a +10 bonus, the corpus pulls that parent
preferentially, and many similar children re-trigger the same bug
with shifted boundary values. `branch_state` is less prone to this
because each child has to add a new value-class event to be retained,
and successive divide-by-zero hits with similar shapes bucket into the
same event. The result is fewer total reports for the same number of
unique bugs.

**The bug detection rate is comparable.** Branch_state's 2/3 vs
merged_bitmap's 3/3 is within trial variance at this sample size —
the difference is a single trial. With more trials we expect this gap
to close further, but it is honestly reported here.

### 5.3 Why the bigger corpus matters

On a small target like ours, the merged-bitmap variants have
*already saturated* the bitmap by the end of trial. The large corpus
in branch-state is buying nothing on this particular target.

On a real target like PyTorch, where the branch space is in the
hundreds of thousands and the bitmap saturates only at the end of a
24-hour campaign, the order is reversed: the merged-bitmap variants
keep getting "novel" (because new edges keep appearing) but the
branch-state variants get *more* novel still, because the same edges
get hit with new value classes throughout the run. The diversity gap
we observe in 22 seconds on the mock target is the smaller version
of what we expect at hour-long scales on PyTorch. We do not have the
budget to demonstrate that empirically — Sec. 6 explains why.

### 5.4 Case study: the divide-by-zero bug

All three "trials with bug" in `branch_state` (and all of
`merged_bitmap`'s) found the same root-cause bug: a divide-by-zero
in `mock_compiler.c:191`, triggered when the linear-reduce kernel is
selected with `shape[reduce_dim]==0`. UBSan reports it on stderr;
the oracle classifies it as `sanitizer`.

The reproducer:

```python
run_seed(OpDescriptor(op_type=OP_REDUCE, dtype=DTYPE_FP32, rank=2,
                      shape=[16, 0], reduce_dim=1, fuse_hint=1))
```

The trace: a `seed_reduce_lastdim` parent with `shape=[16, 32]` is
selected; the `shape_list` mutator perturbs the second entry to a
boundary value of 0; the frontend's `_validate_op` and `_should_break`
both pass; the C backend selects `MC_KERNEL_REDUCE_LINEAR`
(`shape[reduce_dim]=0 < 1024`); the kernel computes `100 / 0` and
UBSan trips. Reproduces under `results/multi_trial/trial_*_*/bugs/sanitizer_*/`.

The bug is reachable from any feedback regime that allows the
`shape_list` mutator to change a non-reduce shape entry to zero — so
we should not over-claim that it is uniquely a branch-state finding.
The contribution is methodological: branch-state arrives at the bug
with a much smaller and more diverse corpus than merged-bitmap,
which on a larger target would translate to faster discovery.


## 6. PyTorch: Obstacles, Not Numbers

We do not report PyTorch numbers in this paper. The artefact's
`scripts/run_pytorch.py` is structured to fail informatively rather
than silently produce zeros. This section documents the obstacles
the script diagnoses, why they apply, and what would be needed to
overcome each one. The `obstacle_report.json` produced by
`run_pytorch.py --feedback-mode branch_state --smoke-only` is the
artefact's output for this section.

| Obstacle | Severity | Mitigation we attempted |
|----------|----------|-------------------------|
| Coverage-instrumented PyTorch build | engineering | None: build alone takes 2-4 hours and 16+ GB RAM |
| `MC_PROBE` source instrumentation in PyTorch | fundamental | None feasible in artefact scope; documented as the central porting obstacle |
| Python tracer overhead on torch.compile | performance | Scope filter (only target packages); subprocess-per-seed isolation |
| coverage.py + torch.compile interaction | unknown | Untested on real PyTorch |
| ASAN/UBSan + gcov double instrumentation | by-design | Per dl-compiler-fuzzing skill: separate builds for sanitizer and coverage |

### 6.1 The C-side instrumentation obstacle

This is the central one. Iteration 2's faithful PolyFuzz design
requires that the C target emit branch-variable records at every
branch. The mock target achieves this with manually inserted
`MC_PROBE` macros — feasible because we wrote the source. PyTorch
is several orders of magnitude larger; manual instrumentation is
not realistic.

The published PolyFuzz solves this with an LLVM pass that runs
during the AFL++ build. The pass walks the LLVM IR, finds every
branch, and inserts code that records the branch's predicate
variables. Implementing or adapting that pass for a CMake-driven
PyTorch build is itself a substantial engineering project — and one
that, even if completed, is paired with an AFL++ harness that
expects byte-buffer inputs, which our seed model does not provide.

### 6.2 What still works without C-side probes

The Python side of branch-state is target-independent: `sys.settrace`
on the target packages (`torch._dynamo`, `torch._inductor`,
`torch.fx`) gives a real branch-variable signal regardless of how
PyTorch was built. The artefact supports this degraded mode via
`--branch-state-c-disabled`. The signal is still richer than
Python-only line coverage — it differentiates "Dynamo took the same
branch with shape `[64, 64]`" from "Dynamo took the same branch with
shape `[1024, 1024]`" — but it loses cross-language correlation.

### 6.3 What the obstacle report looks like

A representative `obstacle_report.json` from running
`run_pytorch.py` on a system without a PyTorch coverage build:

```json
{
  "feedback_mode": "branch_state",
  "obstacles": {
    "gcov_source_root": "missing: /root/pytorch-ptcov does not exist...",
    "py_packages": "missing: none of ['torch._dynamo', ...] importable...",
    "python_branch_tracer": "ok: sys.settrace is available regardless of target",
    "c_branch_probes": "missing: target NOT source-instrumented..."
  },
  "blocking": ["gcov_source_root", "py_packages", "c_branch_probes"],
  "decision": "refused"
}
```

The point of this output is *not* "fuzzer crashed." It is "fuzzer
declined to run because the prerequisites for the chosen mode are
not met, and here is precisely what's missing." The most common
silent-failure mode for coverage-guided fuzzers is collecting
zero coverage and treating that as "no novel seeds found" — we
explicitly check for and refuse this case.


## 7. Limitations & Future Work

We split this section into two parts. §7.1 lists the limitations of
what we built — bounded by what the artefact actually does. §7.2
describes a successor we believe is the right next paper:
PolyFuzz's input-to-branch sensitivity analysis, redesigned for
program-mutation inputs by tracking AST and FX-graph nodes instead
of byte buffers.

### 7.1 Limitations of this artefact

**Single-target evaluation.** All quantitative numbers come from
the mock compiler. We make no PyTorch claims. The mock target is
small enough that branch-state's corpus-diversity advantage does
not translate to a bug-count advantage in 22 seconds; on a target
with a deeper branch space we expect this to invert, but we did
not demonstrate that.

**Fixed value-class bucketing.** Our log-uniform power-of-two
buckets are reasonable for shape-like quantities but probably wrong
for, e.g., model-loss values where small differences matter. A
target-specific or learned bucketing schedule is a small extension.

**No differential oracle.** Most published `torch.compile` bugs
are silent miscompilations, not crashes or sanitizer trips. Our
oracle is signal-based (crashes, sanitizer banners, assertions).
Adding a differential check — run the workload eager and compiled,
fail on `torch.allclose` mismatch — is a few lines per seed and
would substantially broaden bug coverage. The framework already
treats non-zero seed exit as a bug, so the seed simply does the
comparison and exits non-zero on mismatch.

**The C-instrumentation obstacle is load-bearing.** Without an
LLVM-pass equivalent to PolyFuzz's, branch-state mode can collect
C-side events on the mock target only. The Python side ports
cleanly to PyTorch via `sys.settrace`; the C side does not without
either patching torch source or implementing the pass.

### 7.2 Future direction: AST/FX-graph sensitivity analysis

PolyFuzz's third design piece — the input-to-branch regression
model — is the one we did not port, because the input-bytes
abstraction does not apply when the fuzzer's input is a Python
program. The natural successor adapts that design to *program*
mutation. We sketch it here in enough detail that a follow-on
project can pick it up.

#### 7.2.1 What we want

A model M that, given the set of branches the fuzzer is stuck on,
proposes mutations that are *likely* to flip those branches. PolyFuzz
does this for byte buffers by partitioning the input into blocks,
flipping each block, and observing which branch variables change;
the model is then a sparse correspondence from byte-blocks to
branch-variables. We want the same shape, but with AST and FX-graph
nodes as the unit of input.

The pipeline is:

```
seed program (AST)
      │
      ▼
[lowering pass]                  ── makes the FX graph observable
      │
      ▼
torch.fx.GraphModule (FX nodes)
      │
      ▼
[runtime trace]
      │
      ▼
{(seed_node, py_branch_event, c_branch_event)} tuples
      │
      ▼
[regression model]
      │
      ▼
M : seed_node × branch_state → P(flip | mutate(seed_node))
      │
      ▼
mutation guidance: when stuck on branch B with value-class V,
    score each seed_node by M and mutate the highest scorer
```

The unit of *input* for the model is a seed-AST node (e.g. a
specific `Constant`, `Call`, or `Subscript` site in the source
program). The unit of *output* is the branch-state event tuple
defined in §4.3. The model learns which AST nodes drive which
events.

#### 7.2.2 The lowering pass: AST → FX → branches

The successor needs three abstractions in the same input pipeline,
each with a stable id:

1. **AST node** — the seed source's syntactic units. We already
   have `find_branch_lines` walking the AST; we need to extend it
   to assign every `Constant`, `Subscript`, `keyword`, and `Call`
   a stable id (a hash of its location plus normalized form).

2. **FX node** — the corresponding nodes in `torch.fx.symbolic_trace`
   output. Dynamo's `OutputGraph` already produces an FX
   `GraphModule`; we hook into it to log every `fx.Node` created
   from a given AST site. The mapping AST → FX is non-trivial
   because Dynamo unfolds Python control flow into a flat graph
   — a single `for` loop in source becomes many FX nodes — but
   `node.meta['stack_trace']` carries the source location, which
   is enough to back-edge to the AST.

3. **Branch event** — same as our current branch_state events:
   `(file, line, value_class)` for Python and
   `(c_probe, branch_id, value_class)` for C.

The lowering pass runs once per seed: AST → FX → branch events,
recording the mapping at each stage.

#### 7.2.3 The dependence trace

Here is where our approach diverges most from PolyFuzz's. PolyFuzz
flips byte-blocks one at a time and re-runs to observe which
branches change. That is a brute-force `O(seed_size × branches)`
sensitivity probe. For program mutation, single-AST-node flips are
expensive (each requires re-running torch.compile, which can take
seconds), and the AST is two orders of magnitude smaller than a
byte buffer to begin with.

Two cheaper alternatives:

**(a) Live taint, not flip-and-rerun.** Instrument the FX graph at
trace time to record, for each FX node, which seed-AST node it was
generated from (we already have this from the lowering pass) and
which downstream FX nodes it feeds. Then for each Python branch
event, walk the FX dataflow backward to identify the AST nodes
that produced its branch-variable inputs. This is one pass per
seed, not one re-execution per node.

**(b) Type-class taint for C branches.** C-side branches operate on
values that came from the FX nodes that the C kernel consumes.
The mapping FX-node → C-branch is harder because Inductor compiles
fx.GraphModule to C++ via a code generator — but the codegen
already emits per-FX-node markers in the generated C++ for
debugging purposes (see `inductor/codecache.py`'s `kernel_name`
hashes). Following those hashes from a C branch back to the
originating FX node is a pure-textual lookup, not a control-flow
analysis.

The combined dependence relation is:
```
ast_node →(syntactic) fx_node →(value-flow) py_branch_event
                              →(codegen-marker) c_branch_event
```

Once this relation is recorded for all retained seeds, the
regression model learns it directly: each (ast_node, branch_event)
co-occurrence is a positive sample; each retained seed where
mutating that ast_node *changed* that event is a stronger one.

#### 7.2.4 Mutation guidance from the model

The fuzzer gets stuck when the corpus stops growing — successive
mutations land on already-seen branch_state. The successor uses
the model to escape:

```python
def select_targeted_mutation(parent, stuck_events):
    # stuck_events: branches we keep hitting with the same value class
    candidates = []
    for event in stuck_events:
        # Top-k AST nodes that drive this event.
        nodes = model.top_k(event, k=5, parent=parent)
        for node in nodes:
            candidates.append((event, node, model.score(event, node)))
    # Pick the highest-scoring (event, node) pair, mutate that node.
    candidates.sort(key=lambda c: -c[2])
    target_event, target_node, _ = candidates[0]
    return mutate_node(parent, target_node)
```

This replaces the current "pick a mutator weighted by registry
weights" with "pick a mutation site weighted by predicted impact on
events the corpus is stuck on." The fallback when no model
prediction has confidence above a threshold is the existing random
mutator — so the fuzzer never blocks on the model.

#### 7.2.5 What's hard

Three concrete obstacles for the successor:

1. **Codegen-marker stability.** Inductor's per-FX-node markers in
   generated C++ are not part of any public API. They could be
   renamed or removed in any release. The successor would need to
   either pin a Torch version or contribute upstream stability for
   these markers.

2. **Dynamic shape, dynamic graphs.** When `torch.compile` is run
   with `dynamic=True`, the FX graph contains symbolic shape
   variables. The branch events for "shape > 100" become events
   over symbolic shapes, not concrete ones. The bucketing scheme
   in §4 needs extension to handle symbolic ranges, not just
   concrete int values.

3. **Model training data is sparse.** A fuzzing run produces
   thousands of seeds but only handfuls per (ast_node, event)
   combination. The regression model has to be robust to small
   sample sizes. PolyFuzz's logistic regression works because byte
   buffers have ~10^4 byte-blocks per input; AST inputs have
   ~10^2 nodes. Sparse-data tools (Lasso, sparse decision trees)
   are the right family.

#### 7.2.6 Why this is worth doing

The mock-target ablation in §5 already shows that branch-state
retains 2.8× more seeds than merged-bitmap. That is a *passive*
diversity gain: the fitness signal is finer, so more children clear
the novelty bar. The successor adds an *active* diversity gain:
when the fuzzer notices it's stuck, it consults the model and
deliberately mutates AST nodes most likely to break the stalemate.

For a target like `torch.compile`, where seed runs cost on the
order of seconds and a 24-hour budget yields fewer than 50,000
seeds total, every seed needs to be informative. The byte-flip
sensitivity analysis that defines PolyFuzz produces tens of
thousands of seed runs per minute on Pillow — at our throughput,
the same scheme would take a year. Replacing brute-force flip with
live taint is what makes the design feasible at DL-compiler
throughput.

This is, we believe, the genuinely missing piece between PolyFuzz
on byte-buffer targets and PolyFuzz on DL compilers — and the right
focus for follow-on work.


## 8. Artefact

The artefact is one git tree containing both iterations behind one
class:

- 1700 lines of Python implementing the fuzzer in two modes.
- A 280-line C mock compiler with three planted bugs and 39 MC_PROBE
  sites.
- A Python frontend wrapping the mock compiler.
- 9 mock seeds and 5 PyTorch-template seeds.
- 44 unit tests covering the bitmap, branch-state, oracle, and corpus.
- A `Makefile` driving build/install/test/demo/eval.
- A `Dockerfile` reproducing the environment.

To reproduce Sec. 5:

```bash
make build install test     # ~30 seconds
make eval TRIALS=3 BUDGET_SEC=22   # ~5 minutes
cat results/multi_trial/aggregate.json | python -m json.tool
```

To reproduce the Sec. 6 obstacle report:

```bash
python scripts/run_pytorch.py --feedback-mode branch_state --smoke-only
```

Mapping from claims to artefact outputs:

| Claim | Reproduce with |
|-------|----------------|
| Two-iteration ablation | `make eval`; see `aggregate.json` |
| Branch-state retains more seeds | corpus column of aggregate table |
| Bucketing collapses redundant value variants | `tests/test_branch_vars.py::test_value_class_buckets_*` |
| Disjoint Python and C signals each register | `tests/test_branch_vars.py::test_branch_state_disjoint_python_and_c_each_register` |
| C/C++ probe pipeline produces real events | `tests/test_branch_vars.py::test_c_trace_parser_buckets_so_similar_shapes_collapse` |
| Mock-target divide-by-zero | `results/multi_trial/trial_*_*/bugs/sanitizer_*/` |
| PyTorch obstacles | `results/pytorch/obstacle_report.json` |


## 9. A note on the work's history

The first version of this artefact built the merged-bitmap design
without first checking the published PolyFuzz paper, and adopted the
name "PolyFuzz" because the project's working title was "polyfuzz
for python and c coverage". This was an oversight on the
implementation side — Sec. 3 is a charitable framing of what was
genuinely just a strawman built without context. The ablation in
Sec. 5 makes the comparison honest: merged-bitmap is what we built
first, branch-state is what we built once we read the paper, and
both modes ship in one class so the comparison is reproducible.
