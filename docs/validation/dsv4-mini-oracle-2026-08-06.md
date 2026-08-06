# DeepSeek V4 Mini-Oracle Validation — 2026-08-06

## Scope

This report validates the first FP32 mini-oracle milestone. It covers the reduced DeepSeek V4 graph, deterministic synthetic initialization, full-sequence versus incremental decode parity, hash and learned MoE routing, mHC, sliding/CSA/HCA attention state, MTP, and compact oracle generation. It does not validate native FP8/FP4 codecs, C/Vulkan execution, SSD streaming, or the production checkpoint.

## Environment

- Python: 3.13.5
- PyTorch: 2.10.0+cpu
- Platform: Linux x86-64, glibc 2.41
- PyTorch intra-op threads during tests: 1 (set by `tests/python/conftest.py`)
- Pytest plugin isolation: `tools.dsv4_mini.run_tests` disables all external plugin autoloading

## Verification commands

```bash
python -m compileall -q tools/dsv4_mini tests/python
python tools/dsv4_mini/run_tests.py
python -m tools.dsv4_mini.make_oracle \
  --config tools/dsv4_mini/tiny_config.json \
  --output tests/fixtures/dsv4-mini/oracle.json \
  --device cpu \
  --sequence-length 130
```

## Results

- Tests: 24 passed in 26.09 seconds after applying the delivery overlay to a clean simulated checkout.
- Peak resident memory for the post-install test command: 1,215,584 KiB.
- Model parameters: 1,100,290.
- FP32 parameter storage: 4,401,160 bytes.
- Oracle size: 72,221 bytes.
- Oracle SHA-256: `64cb0260b3105beb5c42799c31dee6371a272f0e329802459b5643b7da0d9c6d`.
- Main full/incremental maximum absolute error: `2.1606683731079102e-07`.
- MTP full/incremental maximum absolute error: `2.980232238769531e-07`.
- Main argmax parity: exact.
- MTP argmax parity: exact.
- Route/index parity: exact.

The canonical greedy token sequence is intentionally non-linguistic:

```text
3 20 37 54 71 88 105 188 82 209 67 18
```

## Boundary coverage

The suite checks:

- CSA emission at the rate-4 boundary;
- raw sliding-window transitions at positions 15, 16, and 17;
- HCA emission across positions 126, 127, 128, and 129;
- identical routed-expert selections between full and incremental execution;
- MTP execution through its own learned-routing decoder block and the shared embedding/head.

## Remaining hardware checks

The RTX 2060 Super CUDA command is documented in `tools/dsv4_mini/README.md`. That run was not performed in this CPU-only environment. The BC-250 Vulkan path belongs to a later milestone and is not represented as complete here.
