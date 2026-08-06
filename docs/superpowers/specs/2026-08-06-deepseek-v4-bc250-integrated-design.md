# DeepSeek V4 Flash on AMD BC-250 — Integrated Design

Date: 2026-08-06
Status: approved
Target repository: `matteorossini997-lab/deepseek-v4-in-c`

## Decision

Keep the K3-in-C C99, Safetensors, out-of-core and correctness foundations;
port the Doctor -> Autotune control plane, Vulkan runtime, adaptive cache,
telemetry and quality gates from `q36-bc250-lab`; replace the Kimi-specific
model layer with a new DeepSeek V4 Flash graph and state implementation derived
from the official model repository.

Every source-derived change follows this order:

1. review the owning repository at a recorded commit;
2. inspect implementation, tests, license and notices;
3. classify the candidate as KEEP, PORT, REWRITE, EXPERIMENT, REJECT or DEFER;
4. define an oracle and failure behavior;
5. implement through small, testable commits;
6. let Doctor prove capabilities and Autotune select strategies;
7. preserve raw evidence and provenance.

## Current baseline

The target repository currently exposes only `main` at
`85ab2cd901aa81b70caac7711f06864d594b8ff3`, identical to the upstream K3
commit. It still contains the Kimi K3 CPU-only runtime, KDA/Gated MLA graph,
Kimi tokenizer, `k3_*` namespace and K3 documentation. No DeepSeek support is
implemented yet.

## Target model contract

The DeepSeek layer must be driven by validated checkpoint configuration rather
than duplicated constants. The reference architecture includes:

- 43 layers and hidden size 4096;
- 256 routed experts, top-6 selection and one shared expert;
- three hash-routing layers followed by learned `noaux_tc` routing using
  `sqrtsoftplus`;
- FP8 trunk weights and FP4 experts with explicit scale geometry;
- compressed sparse attention with alternating compression ratios, sliding
  window and sparse index selection;
- Hyper-Connections with `hc_mult=4` and Sinkhorn normalization;
- YaRN/long-context parameters;
- one next-token-prediction layer.

The config parser, tensor inventory, shape checks, format checks and profile
identity together form the model contract. Missing or inconsistent required
fields are hard errors.

## Architecture

```text
K3 foundations
  config discipline | Safetensors | aligned I/O | packing | tiny oracle
  trace replay | cgroup benchmark | portable C99 tests
                         |
                         v
DeepSeek model layer
  config | tensor binding | FP4/FP8 | routing | MoE | sparse attention
  Hyper-Connections | incremental state | MTP
                         |
                         v
BC-250 execution/control
  Vulkan tensor runtime | dynamic memory planner | adaptive multiclass cache
  Doctor -> Autotune | profiler | capture | quality and release gates
```

### Model layer

Owns DeepSeek configuration, tensor names, graph scheduling, routing,
attention, Hyper-Connections, state and MTP. It must not embed storage policy or
Vulkan allocation details.

### Storage layer

Owns Safetensors indexing, validated descriptors, aligned reads, packed trunk
and expert artifacts, synchronous/asynchronous I/O backends and byte-level
metrics. It knows tensor descriptors but not mathematical routing policy.

### Cache layer

Owns route planning, slot state, admission/eviction, classes, trace recording
and counters. It does not know shader internals or checkpoint tensor names.

### CPU backend

Provides the readable deterministic oracle. It is optimized only after
correctness and remains the fallback/reference for every Vulkan operator.

### Vulkan backend

Owns device tensors, views, command lifetime, memory snapshots, scratch pools,
pipelines and profiling. Activations and state remain device-owned across
prefill/decode where possible.

### Control plane

Doctor reports observations and capability gates. Autotune generates candidates
only from proven capabilities, executes one candidate at a time under thermal
supervision, applies correctness and quality gates, and writes a profile tied to
model, pack, GPU, driver and kernel identity.

## Reuse decisions

### KEEP from K3

- one-pass Safetensors scanner and independent verification pattern;
- config fail-closed discipline;
- binder plan that resolves and validates before reading;
- aligned-window O_DIRECT with buffered fallback;
- coalesced expert reads and per-layer trunk packing;
- pinned-prefix trunk design;
- tiny model oracle, PyTorch reference and adversarial fixtures;
- route traces, LRU/Belady replay and memory split harnesses;
- GCC/Clang, Werror, sanitizer and weightless CI gates.

### Preserve as invariants

- slot state includes `INFLIGHT`;
- batch cache work follows reserve -> read -> publish;
- identity is published only after successful complete I/O;
- miss reads are sorted by shard/offset independently from mathematical order;
- arena base and slot stride satisfy direct-I/O alignment;
- cache capacity is at least one unique routed working set;
- trace records model requests before cache outcomes;
- file descriptors start at `-1` and ownership is explicit;
- ring depth greater than one requires a real asynchronous producer and measured
  overlap;
- ring sizing excludes already pinned layers and converges with pin count;
- tensor shape/dtype plans are side-effect-free before reads;
- floating-point reduction order and FMA behavior are part of the reference
  contract;
- dropped or unreadable experts make the run fail distinctly.

### PORT/GENERALIZE from q36

- BC-250 Doctor and temperature-aware Autotune;
- live Vulkan/GTT/TTM memory snapshots and capacity checks;
- private device-local scratch and allocation backoff;
- tensor-resident Vulkan runtime and per-op/per-kernel profiling;
- adaptive cache v4, bounded-heap victim selection and multiclass budgets;
- persistent route bias without blind preload;
- frontier benchmark, quality scoring, BC-250 capture and release QA;
- typed state-codec/checkpoint framework after native DeepSeek state is correct.

### REWRITE

- Kimi config, graph, tensor names, tokenizer and chat template;
- KDA/Gated MLA, Attention Residuals and Stable LatentMoE;
- MXFP4 assumptions not verified against DeepSeek FP4;
- Qwen-specific Vulkan graph and TurboQuant bitstream;
- model state, sparse indexer, Hyper-Connections and MTP.

## Workstreams and gates

1. **WS0 — Baseline and provenance**: status banner, source register, approved
   design, contribution gate and clean baseline.
2. **WS1 — Config and inspector**: fail-closed DeepSeek config, shard inventory,
   tensor/format/profile validation and working-set report.
3. **WS2 — CPU oracle**: ordinary-PyTorch reference, adversarial deterministic
   initialization, tiny isomorphic model, teacher forcing and incremental parity.
4. **WS3 — Storage/packing**: DeepSeek dtypes, manifest, audit, aligned reads and
   I/O backend interface.
5. **WS4 — Vulkan primitives**: tensor runtime, memory/profiler, FP8 and packed
   FP4 operators with CPU parity.
6. **WS5 — Routing/MoE/cache**: hash and learned routing, full-route reservation,
   adaptive multiclass cache, LRU A/B and trace replay.
7. **WS6 — Sparse attention/HCA/state**: indexer, compression, Sinkhorn,
   chunked prefill and complete incremental state.
8. **WS7 — Doctor**: BC-250, model, pack, storage and memory capability gates.
9. **WS8/WS9 — Autotune and evidence**: candidate matrix, thermal supervision,
   quality/performance selection and capture archives.
10. **WS10 — State store/MTP**: typed complete checkpoints, corruption rejection,
    target verification, rollback and measured acceptance.
11. **WS11 — Product surface**: tokenizer/template, sampling, serving and agent
    only after state isolation is proven.

## Doctor -> Autotune contract

Doctor observes hardware, driver, thermals, memory, storage, model and pack.
It emits capability gates; it never silently unlocks an unproven mode. Autotune
then explores only legal candidates across residency split, cache policy,
headroom, I/O backend, ring depth, worker counts, kernel variants, context and
MTP.

A candidate is accepted only when:

```text
model_contract_pass
&& tiny_oracle_pass
&& resident_streaming_parity_pass
&& quality_pass
&& no_oom_or_device_loss
&& thermal_peak_below_limit
&& minimum_frontier_ratio_pass
&& replicated_geometric_mean_gain_pass
```

Memory savings alone never select a codec or strategy.

## Test strategy

- parser/config malformed and mismatch tests;
- adversarial CPU op fixtures;
- tiny teacher-forced, greedy, incremental, chunked and save/restore gates;
- real-layer and selected-logit conformance;
- shader compile, Lavapipe smoke and CPU/Vulkan parity;
- resident/cold/warm/minimal/pressure/failed-read streaming matrix;
- Doctor mock-root and Autotune timeout/thermal/profile tests;
- teacher-forced NLL/logprob/top-k quality comparison;
- typed state corruption, truncation and model/profile mismatch rejection;
- repeated performance runs with raw provenance and hardware-pending status.

## Milestone order

```text
P0 baseline/contract
 -> P1 CPU oracle
 -> P2 storage correctness
 -> P3 Vulkan primitives
 -> P4 MoE streaming
 -> P5 attention/HCA/state
 -> P6 Doctor
 -> P7 Autotune/quality/capture
 -> P8 state store/MTP
 -> P9 serving/product
```

No performance benchmark decides model semantics before the CPU oracle. No
Autotune profile is valid before routing, attention and state parity. No server
precedes state isolation and cancellation tests.

## First release definition

The first useful release requires a validated DeepSeek model contract, exact
tiny CPU model, real-layer reference, CPU/Vulkan parity, resident/streaming
quality parity, Doctor-derived memory planning, replicated Autotune selection,
complete incremental save/restore, green portable/sanitizer/Vulkan CI and raw
BC-250 evidence for every performance claim.
