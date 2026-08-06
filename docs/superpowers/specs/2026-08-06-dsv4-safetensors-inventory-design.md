# DeepSeek V4 Safetensors Inventory Design

Date: 2026-08-06  
Milestone: P0-B2  
Status: implementation design

## Purpose

P0-B2 establishes a fail-closed, inspection-only description of a DeepSeek V4
Flash checkpoint before any weight is loaded or executed. The output becomes the
single source of truth for later binder, Doctor, Autotune, Vulkan residency and
SSD-streaming decisions.

The milestone consumes the Hugging Face `model.safetensors.index.json` and the
JSON header at the front of every referenced Safetensors shard. It never maps,
reads or decodes tensor payloads.

## Reviewed sources

- target `main@94188f6a2abd901459ce35360b46d9fbc2d130bc`;
- K3 Safetensors scanner, header, tests and independent Python verifier inherited
  from `FareedKhan-dev/kimi-k3-in-c@85ab2cd901aa81b70caac7711f06864d594b8ff3`;
- official `deepseek-ai/DeepSeek-V4-Flash` repository head
  `60d8d70770c6776ff598c94bb586a859a38244f1`;
- official model contract revisions `config.json@fd53f94`,
  `inference/model.py@2b2bebc`, `inference/kernel.py@553034d` and
  `inference/convert.py@2b88d47`;
- immutable checkpoint/index revision
  `7d0cce91a2ba9b7738b3b33163a53eccb1cbe47c`;
- Safetensors format rules: the data buffer is fully indexed, tensor spans may
  not overlap, and duplicate JSON keys are invalid.

The official checkpoint revision currently references 46 shards. The
implementation does not hardcode that count: it derives the shard set from the
index so a changed or locally converted checkpoint fails or reports its actual
layout rather than masquerading as the reviewed artifact.

## Relationship to the mini-oracle

The FP32 mini-isomorphic oracle supplies canonical architecture terminology and
later parity gates for hash/learned top-6 routing, mHC/Sinkhorn, sliding/CSA/HCA
state and MTP. It is not a source of production tensor names, dtypes, offsets or
sizes. P0-B2 derives those only from checkpoint metadata.

## Public API

`include/dsv4/dsv4_inventory.h` exposes:

- `dsv4_inventory_open` / `dsv4_inventory_close`;
- O(1) `dsv4_inventory_find`;
- `dsv4_inventory_validate_flash` against a previously validated `DSV4Config`;
- deterministic JSON and TSV writers;
- shard, tensor, dtype, class and byte-total records.

The legacy K3 reader is unchanged. DeepSeek inspection is a separate module so
Kimi behavior and its established tests remain stable during migration.

## Parsing and safety contract

The parser is strict C99 and bounded:

- index files are limited to 128 MiB and shard headers to 1 GiB;
- JSON nesting is limited to 64 levels;
- tensor names are limited to 4095 bytes;
- integer, allocation, element-count, bit-count and aggregate-byte arithmetic is
  overflow checked;
- shard names must be basenames ending in `.safetensors`; absolute paths,
  separators, dot prefixes and `..` are rejected;
- all required fields must be present exactly once;
- unknown root or tensor descriptor fields are rejected;
- ranks above eight and unsupported dtypes are rejected;
- every index entry must appear exactly once in its mapped shard and every shard
  tensor must appear in the index;
- descriptor byte counts must equal shape × dtype bit width;
- spans must begin at byte zero, be contiguous, non-overlapping and consume the
  complete shard data buffer;
- `metadata.total_size` must equal the sum of descriptor bytes.

The dtype table includes current Safetensors scalar formats needed to inspect
modern checkpoints, including sub-byte F4/F6 and FP8/E8M0 variants. Sub-byte
shapes must still resolve to a whole number of bytes.

## Flash profile validation

The structural inventory can be opened without a model config. Passing a
validated `DSV4Config` enables the exact Flash quantization checks:

- 43 base layers plus one MTP layer and expert/layer IDs within range;
- 256 routed experts and learned routing maps whose last dimension is top-6;
- integer routing-index dtypes;
- dense FP8 weights paired with E8M0 scales using 128×128 block geometry;
- routed/shared/MTP FP4 weights paired with E8M0 scales using 32-value K blocks;
- support for native Safetensors `F4` and for the official checkpoint's packed
  `I8` representation, where one physical byte stores two logical FP4 values;
- rejection of missing, orphaned or mismatched quantization scales.

This validates storage geometry, not numerical decode correctness. Bitstream
conversion and GEMM parity belong to later CPU/Vulkan milestones.

## Tensor classes and working-set report

Each tensor receives deterministic metadata:

- dtype, rank, dimensions, shard, relative and absolute offsets, byte count;
- parsed layer and routed-expert IDs where present;
- class: embedding, head, norm, hyper-connection, attention,
  compressor/index, router/hash, routed expert, shared expert, MTP or other.

The inventory aggregates count and bytes by dtype and class. These values are
measured inputs for Doctor and Autotune; no residency percentage or active-byte
estimate is embedded in P0-B2.

## Verification strategy

The C unit test generates valid and malformed shard/index pairs and covers:

- valid lookup, classification, profile validation, JSON and TSV;
- missing or incorrectly mapped shards;
- unsafe paths and duplicate names;
- gaps, overlaps, truncation/byte mismatch and unsupported dtypes;
- malformed or overlong layer/expert/tensor names;
- invalid FP8 and packed-FP4 scale geometry.

A separate Python fixture generator creates deterministic synthetic shard
headers. `tools/verify_dsv4_inventory.py` reparses those headers independently
with Python's JSON implementation and compares every C-emitted tensor on shard,
dtype, shape, absolute offset and byte count.

Required gates are `-Werror`, GCC/Clang, ASan+UBSan, GCC `-fanalyzer`, Make,
CMake/CTest and the independent Python comparison.

## Explicit non-goals

P0-B2 does not:

- download or commit the real checkpoint;
- decode FP4/FP8 values;
- bind tensors to the production graph;
- execute the mini-oracle or production inference from checkpoint weights;
- allocate Vulkan resources;
- choose a cache budget or SSD streaming policy;
- claim validation of the real 46-shard artifact until its headers are run
  through the tool on a machine that has the checkpoint.
