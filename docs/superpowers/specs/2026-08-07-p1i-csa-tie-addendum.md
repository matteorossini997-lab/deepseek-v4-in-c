# P1-I deterministic CSA exact-tie contract

P1-I preserves the P1-D portable index ordering rule: candidates are ranked by score descending and, only when scores are exactly equal, by lower compressed-token index. The official DeepSeek V4 Flash inference path uses PyTorch `topk` and does not specify a cross-backend order for exact ties.

A 129-token diagnostic exposed the first model-level discrepancy at layer-1 CSA position 16: C and Python had identical state counters and the same single non-zero score, while three candidates were exactly zero. The CPU PyTorch backend selected a different tied candidate than the deterministic C implementation.

The mini-oracle therefore uses stable descending `argsort` for index selection. Stable sorting leaves every non-tied ranking unchanged and maps exact ties to their original ascending token order. This is intentionally a reproducibility contract for the project rather than an architectural claim about DeepSeek.

Acceptance evidence: focused tie/model GREEN `31188655524`; full P1-I integration `31189124517` with 25 mini-oracle tests, GCC/Clang, analyzers, ASan+UBSan and CMake/CTest.
