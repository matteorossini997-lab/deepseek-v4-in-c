# P1-K Full Runtime + Greedy Generation — Design Specification

**Date:** 2026-08-07
**Base:** `p1j/native-mtp@b4355d29a2e410ca4555ae8e129b3983f3f07d97`

## Goal

Compose the P1-I base model and P1-J MTP into one transactional batch-one/token-one diagnostic runtime, then add a deterministic greedy generation loop that chooses tokens from the **base-model logits**.

This distinction is deliberate. The reviewed DeepSeek V4 Flash standard inference generator samples from `Transformer.forward` base logits. MTP is a separate module. The project mini-oracle executes base and MTP together for parity diagnostics and also chooses generated tokens from base logits. P1-K preserves both facts: combined state is atomic, while MTP never changes the standard greedy token decision.

## Source review and decisions

- **Target P1-I:** KEEP the base model graph, weight ownership and incremental state. Add only a deep-clone API so a higher-level transaction can stage a token.
- **Target P1-J:** KEEP the standalone MTP graph and shared embedding/head semantics. Add only a deep-clone API.
- **Mini-oracle `MiniReferenceModel.step`:** PORT the combined sequencing: base layers -> base logits -> MTP on final base streams -> one shared absolute-position commit.
- **Mini-oracle `MiniReferenceModel.generate`:** PORT deterministic greedy generation: prefill prompt token-by-token, select `argmax(base_logits)`, append and continue.
- **Official DeepSeek V4 Flash model:** KEEP the distinction that `Transformer.forward` is the base path and MTP remains separate.
- **Official DeepSeek V4 Flash `inference/generate.py` reviewed at source commit `216d76676e09d4e0aa3f73784e9ad53a6d0b2735`:** KEEP base-logit token selection semantics. P1-K uses greedy argmax rather than the official temperature/Gumbel sampling because deterministic greedy parity is the current CPU reference milestone.
- **q36 mini-oracle design branch `d6684ca33d4da496cd2e805ce159c2db7892f98a`:** design input only; no implementation to port.

No official source code or checkpoint bytes are copied.

## Public API

Add deep clone APIs:

- `dsv4_model_clone(const DSV4Model *, DSV4ModelStatus *)`;
- `dsv4_mtp_clone(const DSV4MTP *, DSV4MTPStatus *)`.

Add `dsv4_full_model.h` with:

- `DSV4FullModelConfig { DSV4ModelConfig base; DSV4MTPConfig mtp; }`;
- `DSV4FullModelWeights { DSV4ModelWeights base; DSV4MTPWeights mtp; }`;
- opaque `DSV4FullModel` owning one base runtime and one MTP runtime;
- create/reset/destroy;
- `dsv4_full_model_step_f32` returning base final streams, base logits, MTP final streams, MTP logits and MTP route diagnostics;
- `dsv4_full_model_generate_greedy` for a prompt plus `max_new_tokens`, optional EOS stop, and output sequence/counted length.

The full-runtime validator requires base and MTP dimensions/max sequence to agree and requires MTP shared embedding/head pointers to be identical to the base embedding/head pointers on each step/generation call.

## Whole-model transactionality

A token is one transaction across **both** runtimes:

1. deep-clone base runtime;
2. deep-clone MTP runtime;
3. run base step on the base clone;
4. run MTP step on the MTP clone using the staged base streams;
5. require all base/MTP outputs finite;
6. swap both staged runtimes into the full runtime;
7. publish caller outputs.

Any base or MTP failure discards both clones and leaves both live positions/states plus every caller output unchanged. This is stronger failure atomicity than the Python diagnostic oracle, while preserving successful-path numerics.

## Greedy generation

`dsv4_full_model_generate_greedy` starts from a reset full runtime.

- Prompt length must be at least one and `prompt_len + max_new_tokens <= max_seq_len`.
- Feed every prompt token through the combined step.
- After the final prompt token, choose `argmax(base_logits)` as the next token.
- For each generated token, append it, stop if it equals `eos_id` when EOS checking is enabled, and otherwise feed it through the combined step to obtain the next base logits.
- MTP logits are computed/validated as part of every transaction but never choose the standard generated token.
- The output sequence includes the original prompt followed by generated tokens, matching the mini-oracle greedy test surface.

## Canonical fixture

Create a deterministic full-runtime fixture from the real `MiniReferenceModel` using the same P1-I base fill and P1-J MTP fill.

It records:

- at least 20 successful combined incremental tokens;
- base logits and argmax;
- MTP logits and argmax;
- final base streams and MTP streams where the C API exposes them;
- a prompt of 7 tokens plus 5 greedy generated tokens;
- a late-MTP failure case proving the base state also rolls back.

The fixture may reuse existing P1-I/P1-J weight arrays rather than duplicate weights.

## TDD acceptance

1. RED fixture generation succeeds and linkage fails only on new clone/full-runtime symbols.
2. GREEN combined step matches base/MTP logits and greedy argmax across all fixture positions.
3. Failure injected in MTP after a successful staged base step leaves both runtimes at the same position; repairing the weight and retrying matches the canonical first-token result.
4. Greedy prompt+5 token output exactly matches `MiniReferenceModel.generate`.
5. GCC/Clang, `-fanalyzer`, ASan+UBSan and CMake/CTest pass.
6. Full inherited weightless CI and permanent mini-oracle remain green.

## Non-goals

P1-K does not add tokenizer/chat-template logic, probabilistic/temperature sampling, speculative decoding acceptance, checkpoint binding, FP8/FP4 decode, tensor parallelism, Vulkan or SSD streaming. Those remain subsequent workstreams.
