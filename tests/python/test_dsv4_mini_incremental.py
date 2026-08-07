from __future__ import annotations

from pathlib import Path

import torch

from tools.dsv4_mini.config import load_config
from tools.dsv4_mini.initialization import initialize_reference_model
from tools.dsv4_mini.reference import LearnedRouter, MiniIndexer


CONFIG_PATH = Path(__file__).resolve().parents[2] / "tools" / "dsv4_mini" / "tiny_config.json"


def _tokens(length: int, vocab_size: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.long)
    return ((positions * 17 + 3) % vocab_size).unsqueeze(0)


def _incremental(model, input_ids: torch.Tensor):
    state = model.new_state(batch_size=input_ids.shape[0], device=input_ids.device)
    logits = []
    mtp_logits = []
    traces = []
    for position in range(input_ids.shape[1]):
        output = model.step(input_ids[:, position : position + 1], state)
        logits.append(output.logits)
        mtp_logits.append(output.mtp_logits)
        traces.append(output.trace)
    return torch.cat(logits, dim=1), torch.cat(mtp_logits, dim=1), traces


def test_full_and_incremental_match_through_sliding_and_csa_boundaries() -> None:
    cfg = load_config(CONFIG_PATH)
    model = initialize_reference_model(cfg, device="cpu").eval()
    input_ids = _tokens(18, cfg.vocab_size)

    full = model.forward_full(input_ids)
    logits, mtp_logits, traces = _incremental(model, input_ids)

    torch.testing.assert_close(full.logits, logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full.mtp_logits, mtp_logits, rtol=1e-5, atol=1e-6)
    for position in [3, 4, 15, 16, 17]:
        for layer_idx in range(cfg.num_hidden_layers):
            assert full.trace[position].layers[layer_idx].route_indices == traces[position].layers[layer_idx].route_indices
            assert (
                full.trace[position].layers[layer_idx].attention.compressed_count
                == traces[position].layers[layer_idx].attention.compressed_count
            )
        assert full.trace[position].mtp.route_indices == traces[position].mtp.route_indices


def test_indexer_breaks_exact_score_ties_by_lower_token_index() -> None:
    """P1-D's portable contract is score-descending, then token-index ascending.

    The official PyTorch implementation calls topk directly, whose order for exact
    ties is backend-dependent. The mini-oracle fixes that otherwise unspecified
    case so CPU/Vulkan/CUDA references compare the same selected KV positions.
    """

    cfg = load_config(CONFIG_PATH)
    indexer = MiniIndexer(cfg, rate=4).eval()
    indexer.top_k = 2
    with torch.no_grad():
        indexer.q_proj.weight.zero_()
        indexer.weight_proj.weight.zero_()
    state = indexer.new_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    state.compressed = torch.zeros(1, 4, cfg.index_head_dim)

    hidden = torch.zeros(1, 1, cfg.hidden_size)
    q_residual = torch.zeros(1, 1, cfg.q_lora_rank)
    _, indices = indexer.step(hidden, q_residual, position=16, state=state)

    assert indices == (0, 1)


def test_full_and_incremental_match_hca_boundary_127_to_129() -> None:
    cfg = load_config(CONFIG_PATH)
    model = initialize_reference_model(cfg, device="cpu").eval()
    input_ids = _tokens(130, cfg.vocab_size)

    full = model.forward_full(input_ids)
    logits, mtp_logits, traces = _incremental(model, input_ids)

    for position in [126, 127, 128, 129]:
        torch.testing.assert_close(full.logits[:, position], logits[:, position], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(full.mtp_logits[:, position], mtp_logits[:, position], rtol=2e-5, atol=2e-6)
        assert full.trace[position].layers[3].attention.compressed_count == traces[position].layers[3].attention.compressed_count
        assert full.trace[position].layers[5].attention.compressed_count == traces[position].layers[5].attention.compressed_count
    assert traces[126].layers[3].attention.compressed_count == 0
    assert traces[127].layers[3].attention.compressed_count == 1


def test_mtp_block_uses_learned_router_and_shared_vocabulary_head() -> None:
    cfg = load_config(CONFIG_PATH)
    model = initialize_reference_model(cfg, device="cpu").eval()

    assert isinstance(model.mtp.decoder.moe.router, LearnedRouter)
    assert model.mtp.head is model.lm_head
    assert model.mtp.embed_tokens is model.embed_tokens

    output = model.forward_full(_tokens(5, cfg.vocab_size))
    assert output.mtp_logits.shape == (1, 5, cfg.vocab_size)
    assert len(output.trace[0].mtp.route_indices) == cfg.num_experts_per_token


def test_initialization_and_greedy_generation_are_deterministic() -> None:
    cfg = load_config(CONFIG_PATH)
    first = initialize_reference_model(cfg, device="cpu").eval()
    second = initialize_reference_model(cfg, device="cpu").eval()

    for name, tensor in first.state_dict().items():
        torch.testing.assert_close(tensor, second.state_dict()[name], rtol=0, atol=0)

    prompt = _tokens(7, cfg.vocab_size)
    generated_first = first.generate(prompt, max_new_tokens=5)
    generated_second = second.generate(prompt, max_new_tokens=5)
    assert torch.equal(generated_first, generated_second)
    assert generated_first.shape == (1, 12)
