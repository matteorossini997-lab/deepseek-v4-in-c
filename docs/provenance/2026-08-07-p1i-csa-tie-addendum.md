# P1-I CSA exact-tie determinization addendum

**Review date:** 2026-08-07

## Source review

The DeepSeek V4 Flash inference indexer was re-reviewed before this correction. The official path computes its final compressed-token index scores and calls PyTorch `topk` directly. That defines the selected highest scores but does not provide a portable ordering rule for candidates with exactly equal scores. The P1-D C implementation already had an explicit cross-backend contract: rank by score descending, then by lower compressed-token index for exact ties.

During the 129-token P1-I model-shell gate, the first C/Python divergence was localized to layer-1 CSA at position 16. The two implementations agreed on cache/compression counts, current token, MoE route set and every non-tied score. The index scores were approximately `[0.0059833, 0, 0, 0]`; CPU PyTorch selected a backend-dependent tied candidate while C selected the lower index.

## Decision

**KEEP** the P1-D portable C behavior. **DETERMINIZE** only exact ties in the mini-oracle with a stable descending sort, which preserves ascending original token index for equal values. Non-tied rankings and all DeepSeek scoring equations remain unchanged. This is a local reproducibility rule for CPU/Vulkan/CUDA parity, not a claim that DeepSeek mandates a particular exact-tie order.

## Verification

Focused GREEN run `31188655524` passed the exact-tie regression, the full mini-oracle suite, the P1-D sparse-index regression and the P1-I 129-token model parity. Full P1-I integration run `31189124517` passed 25 mini-oracle tests, GCC and Clang model parity, GCC analyzers, ASan+UBSan, CMake/CTest and tree hygiene.
