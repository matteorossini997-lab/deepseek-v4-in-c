from __future__ import annotations

import torch
from torch import nn

from .config import MiniConfig
from .reference import (
    GroupedLinear,
    HashRouter,
    HyperConnection,
    HyperHead,
    LearnedRouter,
    MiniAttention,
    MiniCompressor,
    MiniReferenceModel,
    RMSNorm,
)


def initialize_reference_model(config: MiniConfig, device: str | torch.device = "cpu") -> MiniReferenceModel:
    """Create deterministic adversarial weights without downloading a checkpoint."""

    torch.manual_seed(config.seed)
    model = MiniReferenceModel(config).to(device)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, GroupedLinear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                module.weight.fill_(1.0)
            elif isinstance(module, HashRouter):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, LearnedRouter):
                nn.init.normal_(module.weight, mean=0.0, std=0.003)
                module.correction_bias.copy_(
                    torch.linspace(-0.45, 0.45, config.num_routed_experts, device=module.correction_bias.device)
                )
            elif isinstance(module, HyperConnection):
                nn.init.normal_(module.fn, mean=0.0, std=0.02)
                module.base.copy_(torch.linspace(-0.1, 0.1, module.base.numel(), device=module.base.device))
                module.scale.copy_(torch.tensor([0.9, 1.1, 0.8], device=module.scale.device))
            elif isinstance(module, HyperHead):
                nn.init.normal_(module.fn, mean=0.0, std=0.02)
                module.base.copy_(torch.linspace(-0.08, 0.08, module.base.numel(), device=module.base.device))
                module.scale.fill_(0.9)
            elif isinstance(module, MiniCompressor):
                module.position_bias.copy_(
                    torch.linspace(
                        -0.06,
                        0.06,
                        module.position_bias.numel(),
                        device=module.position_bias.device,
                    ).view_as(module.position_bias)
                )
            elif isinstance(module, MiniAttention):
                module.sinks.copy_(
                    torch.linspace(-0.2, 0.2, config.num_attention_heads, device=module.sinks.device)
                )

        for layer_idx, layer in enumerate([*model.layers, model.mtp.decoder]):
            for expert_idx, expert in enumerate(layer.moe.experts):
                factor = 0.75 + 0.035 * expert_idx + 0.01 * layer_idx
                for projection in (expert.gate_proj, expert.up_proj, expert.down_proj):
                    projection.weight.mul_(factor)
            for projection in (
                layer.moe.shared_expert.gate_proj,
                layer.moe.shared_expert.up_proj,
                layer.moe.shared_expert.down_proj,
            ):
                projection.weight.mul_(1.15)

    model.eval()
    sample_ids = torch.arange(32, device=model.embed_tokens.weight.device).view(1, -1) % config.vocab_size
    sample_hidden = model.embed_tokens(sample_ids)
    learned_router = model.layers[config.num_hash_layers].moe.router
    if not isinstance(learned_router, LearnedRouter):
        raise RuntimeError("first learned layer did not construct a LearnedRouter")
    logits, _, biased = learned_router(sample_hidden)
    unbiased = torch.topk(
        torch.sqrt(torch.nn.functional.softplus(logits.float())),
        config.num_experts_per_token,
        dim=-1,
        sorted=False,
    ).indices
    biased_sorted = torch.sort(biased, dim=-1).values
    unbiased_sorted = torch.sort(unbiased, dim=-1).values
    if not torch.any(biased_sorted != unbiased_sorted):
        raise RuntimeError("adversarial initialization failed to make correction bias change a route")
    return model
