# P1-G Native MoE Design

**Date:** 2026-08-07

## Goal

Add a portable FP32 batch-one/token-one Mixture-of-Experts reference that matches the canonical mini-oracle `SparseMoE` and preserves the production DeepSeek routing/expert contract closely enough to compose into the next native decoder-layer milestone.

## Source review

Re-reviewed before implementation:

- target P1-F `1106f248ccea9bc3b06cb031e8ca58542e23c8c7`;
- target `tools/dsv4_mini/reference.py`: `HashRouter`, `LearnedRouter`, `Expert`, `SparseMoE`, `DecoderLayer`;
- P1-A `include/dsv4/dsv4_cpu_ops.h` routing API and canonicalization rule;
- official `deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1`, current inference `Gate`, `Expert`, `MoE`, and block order;
- current official inference configuration: sqrt-softplus, route scale 1.5, top-6, one shared expert and SwiGLU limit 10.

## Decision

- **KEEP** P1-A `sqrtsoftplus` and learned correction-bias routing.
- **REWRITE** MoE expert execution and route aggregation in C99.
- **DO NOT** bake the mini-oracle synthetic hash formula into the production-facing MoE API. Hash layers consume an explicit caller-owned `tid2eid` table, matching the official checkpoint contract. The fixture builds the mini-oracle table and passes it through that same API.

No official source file or checkpoint data is copied.

## Public contract

`DSV4MoEConfig` contains hidden/intermediate widths, routed expert count, top-k, hash-layer count, vocabulary size, route scale and SwiGLU limit.

`DSV4MoEWeights` exposes caller-owned FP32 arrays:

- router matrix `[experts, hidden]`;
- correction bias `[experts]` for learned layers;
- hash table `[vocab, top_k]` for hash layers;
- routed expert gate/up `[experts, intermediate, hidden]` and down `[experts, hidden, intermediate]`;
- one shared expert gate/up `[intermediate, hidden]` and down `[hidden, intermediate]`.

`dsv4_moe_forward_f32` consumes one hidden vector, layer id and token id and publishes output plus route indices/weights transactionally.

## Numerical order

For each token:

1. compute router logits and sqrt-softplus scores in FP32;
2. hash mode: gather explicit `tid2eid` indices; learned mode: reuse P1-A correction-bias top-k;
3. normalize selected unbiased scores and multiply by route scale;
4. each selected expert executes `gate = clamp_max(W_gate x)`, `up = clamp(W_up x, -limit, +limit)`, `silu(gate) * up`, then down projection;
5. multiply routed expert output by its route weight and accumulate;
6. execute the shared expert unweighted and add it;
7. publish route pairs and output only after all validation/computation succeeds.

The official inference implementation may multiply route weights before the down projection; because the down projection is linear this is semantically equivalent. The C reference follows the mini-oracle order and validates final FP32 output against it.

## Deterministic fixture

A generator constructs a reduced valid `MiniConfig` (hidden 8, intermediate 6, 8 routed experts, top-2, one hash layer), fills the real `SparseMoE` with explicit binary-fraction weights and correction bias, and emits:

- hash-layer cases with different token ids;
- learned-layer cases with different hidden vectors;
- canonical route expert/weight pairs sorted by expert id for backend-independent diagnostics;
- final hidden outputs.

## Error contract

Missing mode-required arrays, out-of-range token/table entries, invalid dimensions, non-finite input/weights/results and allocation/size overflow fail closed. Caller output and route buffers are unchanged on error.

## Non-goals

No FP4 unpack/dequantization, checkpoint binding, tensor-parallel expert sharding, expert cache/SSD streaming, decoder-layer mHC composition or Vulkan kernels in P1-G.