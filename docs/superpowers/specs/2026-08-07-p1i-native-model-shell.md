# P1-I Native Base Model Shell Design

**Date:** 2026-08-07

## Goal

Compose P1-H decoder layers into the first native FP32 base-model path: token embedding, Hyper-Connection stream expansion, sequential decoder layers, final HyperHead collapse, weighted RMSNorm, and vocabulary LM head. MTP remains a separate P1-J increment.

The claim remains correctness-first: batch size one and one token per incremental step.

## Source review

Re-reviewed before implementation:

- target P1-H `df79a91c4e24bf3629dd4bd6c57557163393a0ce` and its decoder-layer API/tests;
- target mini-oracle `MiniReferenceModel`, `_expand_embeddings`, `HyperHead`, `DecoderLayer.step`, and model state handling;
- official `deepseek-ai/DeepSeek-V4-Flash` current inference model on 2026-08-07. The top-level Transformer path remains embedding -> HC copies -> block stack -> HC head/final norm -> vocabulary head; MTP blocks are separate modules sharing embedding/head;
- current official inference config continues to report 43 base layers plus one MTP entry in `compress_ratios`, vocab 129280, HC multiplier 4 and the already-pinned attention/MoE geometry.

## Decision

- **KEEP** P1-H native decoder layers.
- **REWRITE** only model ownership, embedding lookup/HC expansion, HyperHead collapse, final weighted RMSNorm, dense LM head and multi-layer transactional orchestration.
- **EXTEND** P1-H with a deep-clone API for `DSV4DecoderLayer`. This lets a model token run on cloned layer states and commit all layers only after final logits are valid.
- **DEFER** MTP to P1-J so base-model logits/state parity can be isolated from speculative-head behavior.

No official implementation source or checkpoint data is copied.

## Public contract

`DSV4ModelConfig` contains:

- `vocab_size`, `hidden_size`, `hc_mult`, `num_layers`, `max_seq_len`;
- final HC epsilon and RMS epsilon;
- a caller-owned array of `DSV4DecoderLayerConfig` used only during construction and copied internally.

`DSV4ModelWeights` contains caller-owned immutable pointers:

- token embedding `[vocab_size, hidden_size]`;
- one `DSV4DecoderLayerWeights` entry per base layer;
- HyperHead `fn [hc_mult, hc_mult * hidden_size]`, `base [hc_mult]`, `scale [1]`;
- final RMSNorm weight `[hidden_size]`;
- LM head `[vocab_size, hidden_size]`.

`DSV4Model` owns a deep stateful array of decoder layers and `next_position`.

`dsv4_model_step_f32` consumes one token id and publishes:

- final decoder streams `[hc_mult, hidden_size]` for later MTP reuse;
- base-model logits `[vocab_size]`.

## Exact numerical order

For one token at the model-owned absolute position:

1. gather one embedding row;
2. copy that hidden vector into all `hc_mult` streams;
3. deep-clone every P1-H decoder layer before executing any of them;
4. run cloned decoder layers sequentially with the same token id and position;
5. flatten final streams and unweighted RMS-normalize across `hc_mult * hidden_size`;
6. compute HyperHead logits `fn @ flat`, then `sigmoid(mix * scale + base) + hc_eps`;
7. collapse the original final streams with those pre-weights;
8. weighted RMSNorm the collapsed hidden vector;
9. compute the dense vocabulary LM head;
10. only after all logits are finite, replace every live decoder layer with its successful clone, increment `next_position`, and publish streams/logits.

This mirrors mini-oracle `HyperHead` and the official inference HC head semantics.

## Canonical fixture

A generator instantiates the real mini-oracle `MiniReferenceModel` with three base layers and one unused MTP module. It fills:

- token embedding;
- all three real `DecoderLayer` modules using the P1-H deterministic formulas;
- base HyperHead;
- final weighted RMSNorm;
- LM head;

with explicit binary-fraction weights.

The base path is executed directly through the real model modules, deliberately not calling MTP. A 129-token incremental sequence therefore traverses:

- layer 0 sliding attention + hash MoE;
- layer 1 CSA + learned MoE;
- layer 2 HCA + learned MoE, including its first compressed entry at position 127.

The fixture records final streams, vocabulary logits and argmax tokens for each step.

## Whole-model rollback test

After one valid token, the test makes a **later decoder layer** invalid by removing its learned MoE correction bias. The first cloned layer(s) therefore execute successfully before the failure. Model outputs and original layer states must remain untouched. Restoring the weight and retrying the same token/position must match the canonical fixture.

## Non-goals

P1-I does not execute MTP, tokenizer/chat templates, generation loops, checkpoint binding, FP8/FP4 decoding, tensor parallelism, Vulkan, Doctor/Autotune, or SSD expert streaming.