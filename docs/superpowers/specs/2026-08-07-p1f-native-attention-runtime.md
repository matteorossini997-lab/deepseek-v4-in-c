# P1-F Native Attention Runtime Design

**Date:** 2026-08-07

## Goal

Close the remaining gap between the weightless P1-E post-projection attention composer and the canonical `MiniAttention.step`: own incremental attention state in C99, execute the learned FP32 projections and normalizations, update raw/compressed/index caches, and call the already-verified post-projection graph.

The claim is intentionally narrow: **batch size 1, one token per step, FP32 reference execution**. Full-sequence execution is obtained by iterating the same step, matching the mini-oracle implementation.

## Source review

Before implementation the following sources were re-reviewed:

- target branch `p1e/postprojected-attention-step@70c79636aa92af42552da035290348e9e504ffa0`;
- target `tools/dsv4_mini/reference.py`, especially `MiniCompressor`, `MiniIndexer` and `MiniAttention.step`;
- target `tools/dsv4_mini/initialization.py` and tiny config contract;
- official `deepseek-ai/DeepSeek-V4-Flash` main revision observed as `60d8d70770c6776ff598c94bb586a859a38244f1`;
- official `inference/model.py` attention order: `wq_a -> q_norm -> wq_b -> per-head RMS`, `wkv -> kv_norm`, raw/compressed cache maintenance, optional indexer, sparse attention, inverse RoPE, grouped `wo_a`, dense `wo_b`;
- P1-B/P1-C/P1-D/P1-E C APIs and tests.

A separately discussed `DeepSeek-V4-Flash-0731` release was not substituted for the approved target because no separate indexed inference contract was established during this review.

## Decision

**REWRITE** the runtime ownership and learned-projection glue in portable C99. **KEEP** the verified P1-C compressor/RoPE/sink primitives, P1-D sparse top-k and P1-E post-projection composer. No official implementation source or checkpoint data is copied.

## API and ownership

`DSV4AttentionRuntime` owns all mutable state:

- next absolute position;
- chronological raw sliding KV cache;
- projected compressor window buffers;
- compressor overlap state;
- append-only compressed attention KV;
- CSA indexer projected window, overlap state and compressed index KV.

`DSV4AttentionRuntimeWeights` is caller-owned and immutable. It exposes FP32 pointers for:

- `q_a`, weighted `q_a` RMSNorm, `q_b`;
- shared `kv` projection and weighted KV RMSNorm;
- sinks, grouped `o_a`, dense `o_b`;
- attention compressor projection/gate/position-bias/norm weights when rate is nonzero;
- CSA indexer query/head-weight projections and its own compressor weights when rate is 4.

## Step semantics

For one hidden vector at absolute `position`:

1. reject invalid position, dimensions, non-finite input or required non-finite weights before mutation;
2. clone mutable state into a temporary working state;
3. `q_residual = RMSNorm(q_a(hidden))`;
4. `q = q_b(q_residual)`, then unweighted RMS normalize each head;
5. `kv = RMSNorm(kv(hidden))`, apply partial RoPE and append to the sliding cache;
6. if compressed attention is enabled, project hidden into compressor KV/gate buffers and publish one compressed entry only at a completed window;
7. in CSA mode, run the independent indexer compressor plus `q_proj(q_residual)` and `weight_proj(hidden)`;
8. invoke `dsv4_attention_postprojected_step_f32` with chronological raw KV and the owned compressed arrays;
9. publish hidden output, trace and the cloned state atomically only after all dependencies succeed.

The runtime is correctness-first. Cloning state per token is accepted in P1-F; allocator/performance work belongs to later CPU/Vulkan runtime phases.

## Deterministic parity fixture

A new Python generator constructs a deliberately small but structurally valid `MiniConfig` with sliding, CSA and HCA layers. Every attention parameter is filled with an explicit binary-fraction formula rather than RNG output. The C unit test recreates exactly the same weights and hidden sequence from the same integer formulas.

The fixture records `MiniAttention.step` outputs and traces across:

- sliding-window wraparound;
- multiple CSA compression boundaries and top-k selection;
- the HCA first-compression boundary at position 127 and the following step.

This avoids embedding thousands of weight constants while still executing the real canonical PyTorch classes.

## Error contract

The step is fail-closed. On any error:

- runtime state is unchanged;
- output hidden vector is unchanged;
- selected-index output and trace are unchanged.

Dimension multiplication/allocation sizes are overflow checked. Required weights depend on compression mode; unused optional weights are not dereferenced.

## Non-goals

P1-F does not bind Safetensors tensors, decode FP8/FP4 checkpoint storage, execute MoE/mHC decoder layers, implement batch >1, optimize allocations, implement Vulkan, Doctor/Autotune, or SSD streaming.