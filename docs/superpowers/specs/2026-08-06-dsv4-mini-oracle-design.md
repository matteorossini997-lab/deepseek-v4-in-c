# DeepSeek V4 Flash Mini-Oracle Design

Date: 2026-08-06
Status: approved
Target repository: `matteorossini997-lab/deepseek-v4-in-c`

## Goal

Provide a deterministic, reduced-dimension DeepSeek V4 Flash reference graph that runs on CPU and RTX 2060 Super without production weights or native FP8/FP4 kernels.

## Fidelity boundary

The fixture preserves the model-path categories and state transitions needed by a future C/Vulkan backend: interleaved partial RoPE, shared-KV MQA, grouped output projection, sliding attention, compressed sparse attention at rate 4, heavily compressed attention at rate 128, mHC residual streams, hash and learned top-6 MoE routing, a shared expert, bounded SwiGLU, and one MTP block. Widths, vocabulary, context ceiling, expert count, and repeated layer count are reduced.

This milestone is FP32 only. Native FP8/FP4 packing, the C loader, Vulkan kernels, SSD streaming, and real-checkpoint route locality are follow-on milestones.

## Acceptance

- CPU tests require no checkpoint.
- Fixed seed reproduces weights, routes, logits, checksums, and generated IDs.
- Full-sequence and incremental decode agree at all positions, including 15/16/17 and 127/128/129.
- Hash routes contain six unique experts.
- Learned selection uses `scores + correction_bias`, while selected weights come from uncorrected scores and are normalized before route scaling.
- mHC combination matrices are approximately doubly stochastic.
- The MTP path runs through a learned-routing decoder block and shared vocabulary head.
- Oracle generation fails closed if adversarial fixture checks are not exercised.
