# DeepSeek V4 Attention State and Layout Design

Date: 2026-08-06

## Scope

P1-B introduces the state-machine and tensor-layout primitives required by the
native DeepSeek attention path:

- sliding raw-KV ring metadata;
- CSA compression and indexer emission counts at rate 4;
- HCA compression emission counts at rate 128;
- strict incremental position sequencing;
- shared single-head KV expansion across query heads;
- grouped output projection matching the mini-oracle `GroupedLinear` layout.

This increment deliberately does not implement query/KV projections, RMS norms,
RoPE, compressor pooling, sparse indexing scores, sink softmax, attention value
accumulation or the final output projection chain.

## Canonical source

The behavioral reference is the merged FP32 mini-isomorphic oracle at
`13952eff5dcf75ce75e692d5a03e714e47de2fd4`, especially:

- `MiniAttention`, `MiniCompressor`, `MiniIndexer` and `GroupedLinear` in
  `tools/dsv4_mini/reference.py`;
- `tests/python/test_dsv4_mini_attention.py`.

The relevant canonical boundaries are:

- sliding raw cache grows to `sliding_window` and never exceeds it;
- CSA emits at positions 3, 7, 11, ... and its indexer count matches the
  compressor count;
- HCA emits its first compressed entry at position 127;
- the raw-KV tensor has one shared head before attention fan-out;
- grouped projection computes `...gi,goi -> ...go` without cross-group mixing.

## State-machine contract

`DSV4AttentionState` contains metadata only. Tensor storage belongs to the
future attention implementation and can use the returned ring slot and logical
start position.

For every accepted step:

1. `position` must equal `next_position`;
2. `raw_write_slot = position mod sliding_window`;
3. raw length grows until the window is full;
4. `raw_start_position` identifies the oldest logical token retained;
5. rate 4 and rate 128 increment one pending-token counter;
6. reaching the rate emits exactly one compressed entry and resets the pending
   counter;
7. only rate 4 has a matching indexer emission;
8. state and trace are committed atomically after all overflow checks.

A rejected step must not mutate state.

## Layout primitives

`dsv4_cpu_repeat_shared_kv` expands `[tokens, head_dim]` to
`[tokens, num_heads, head_dim]` by exact copies. It exists as a reference/layout
primitive; a production kernel may broadcast rather than materialize.

`dsv4_cpu_grouped_linear` consumes row-major
`[tokens, groups, in_per_group]` and weights
`[groups, out_per_group, in_per_group]`, producing
`[tokens, groups, out_per_group]`. Each group is independent.

## Safety and numerical rules

- Only rates 0, 4 and 128 are accepted.
- Zero dimensions, non-finite inputs and size overflows are rejected.
- Position `UINT64_MAX` is rejected before increment.
- Shared-KV byte strides are overflow checked before `memcpy`.
- Grouped accumulation rejects non-finite results.
- GCC analyzer, ASan and UBSan are mandatory.

## Acceptance

P1-B is accepted when:

- sliding lengths and ring slots match the canonical window boundaries;
- CSA counts equal `[0,0,0,1,1,1,1,2,2]` for nine steps;
- HCA counts are zero through position 126 and one from position 127;
- invalid/noncontiguous steps leave state unchanged;
- shared-KV copies and grouped projection outputs are exact;
- GCC and Clang, Make, CMake/CTest and all existing workflows remain green.
