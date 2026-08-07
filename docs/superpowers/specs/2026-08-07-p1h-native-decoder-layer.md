# P1-H Native Decoder Layer Design

**Date:** 2026-08-07

## Goal

Compose the verified native primitives into the first complete FP32 batch-one, one-token DeepSeek decoder layer: Hyper-Connection pre-mix, weighted RMSNorm, stateful attention, Hyper-Connection post-mix, second Hyper-Connection pre-mix, weighted RMSNorm, sparse MoE, and final Hyper-Connection post-mix.

## Source review

Re-reviewed before implementation:

- target P1-G `7195f1e3ff54e310ee35fb5eacdee446c31a8c78`;
- target mini-oracle `DecoderLayer.step`, `_mix`, `HyperConnection`, `SparseMoE`, and `MiniAttention.step`;
- P1-A `dsv4_cpu_hyperconnection` contract;
- P1-F native attention runtime API and implementation;
- P1-G native MoE API and tests;
- official `deepseek-ai/DeepSeek-V4-Flash` inference model, current page reviewed 2026-08-07. The current `Block.forward` still executes HC-pre -> attention norm -> attention -> HC-post, followed by HC-pre -> FFN norm -> MoE -> HC-post. The model page currently reports `inference/model.py` commit `2b2bebc`; the approved repository provenance remains rooted in the reviewed DeepSeek V4 Flash source lineage.

## Decision

- **KEEP** P1-A Hyper-Connection numerics.
- **KEEP** P1-F learned-projection/stateful attention runtime.
- **KEEP** P1-G sparse MoE.
- **REWRITE** only decoder-layer ownership, weighted RMS norms, HC post-mix glue, tracing, and transactional orchestration in C99.
- **EXTEND** the opaque P1-F attention runtime with a deep-clone API. This is required so a late MoE failure cannot leave KV/compressor state advanced by half a decoder layer.

No official source or checkpoint data is copied.

## Runtime ownership and transactionality

`DSV4DecoderLayer` owns one `DSV4AttentionRuntime *`. Hyper-Connection and MoE are stateless for inference and consume caller-owned immutable weights.

For each token the layer first clones the attention runtime. Every operation runs against temporary output buffers and the cloned attention state. Only after the second HC post-mix succeeds does the layer:

1. publish output streams and diagnostic trace;
2. destroy the old attention runtime;
3. install the successful clone.

Any failure before that point destroys the clone and leaves the original layer state plus every caller output unchanged.

## Public contract

`DSV4DecoderLayerConfig` contains:

- `layer_idx`, `hidden_size`, `hc_mult`;
- Sinkhorn iterations, HC epsilon, RMS epsilon;
- embedded `DSV4AttentionRuntimeConfig`;
- embedded `DSV4MoEConfig`.

The constructor rejects inconsistent hidden sizes, invalid HC geometry, or an attention/MoE contract incompatible with the layer.

`DSV4DecoderLayerWeights` contains:

- attention HC `fn/base/scale`;
- FFN HC `fn/base/scale`;
- input weighted RMSNorm weight;
- post-attention weighted RMSNorm weight;
- embedded attention runtime weights;
- embedded MoE weights.

`dsv4_decoder_layer_step_f32` consumes one `[hc_mult, hidden_size]` stream group plus token id and absolute position. It publishes `[hc_mult, hidden_size]` streams, attention selected indices/count, MoE route indices/weights, and an attention-count trace.

## Exact numerical order

For one token:

1. `dsv4_cpu_hyperconnection(streams, attention_hc_*)` -> `post`, `comb`, `collapsed`;
2. weighted RMSNorm of `collapsed` using `input_norm_weight`;
3. P1-F attention step on a cloned attention runtime;
4. HC post-mix matching mini-oracle `_mix`: `comb^T @ residual_streams + post * attention_output`;
5. second `dsv4_cpu_hyperconnection` using FFN HC weights;
6. weighted RMSNorm using `post_attention_norm_weight`;
7. P1-G MoE using `layer_idx` and token id;
8. second HC post-mix;
9. atomically publish diagnostics, output streams, and cloned attention state.

## Canonical fixture

A new Python generator instantiates real mini-oracle `DecoderLayer` modules with deterministic binary-fraction weights. It covers:

- layer 0: hash routing + sliding attention through raw-window wraparound;
- layer 1: learned routing + CSA through multiple compression/index boundaries;
- layer 2: learned routing + HCA through the first rate-128 emission at position 127 and the following token.

The C test reconstructs the same attention, MoE, HC and norm weights from integer formulas. It compares final streams, attention counts/indices, and MoE expert/weight pairs. A deliberate non-finite shared-expert weight after valid HC/attention inputs verifies full-layer rollback: fixing the weight and retrying the same absolute position must succeed.

## Non-goals

P1-H does not bind checkpoint tensors, execute embeddings or the LM head, implement MTP, decode FP8/FP4 storage, optimize allocation/copies, provide tensor parallelism, Vulkan, Doctor/Autotune, or SSD expert streaming.