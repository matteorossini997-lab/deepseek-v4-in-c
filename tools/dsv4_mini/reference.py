from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import MiniConfig


def sqrtsoftplus(x: torch.Tensor) -> torch.Tensor:
    """DeepSeek V4 routing score: sqrt(softplus(logit))."""

    return torch.sqrt(F.softplus(x.float()))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(input_dtype)


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)


def _rope_cos_sin(
    positions: torch.Tensor,
    rope_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    frequencies = 1.0 / (
        theta ** (torch.arange(0, rope_dim, 2, device=positions.device, dtype=torch.float32) / rope_dim)
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _apply_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Apply interleaved RoPE to a [B, H, S, D] tensor's trailing slice."""

    if rope_dim == 0:
        return x
    leading = x[..., :-rope_dim]
    rope = x[..., -rope_dim:]
    pairs = rope.view(*rope.shape[:-1], rope_dim // 2, 2)
    cos = cos[:, None, :, :, None]
    sin = sin[:, None, :, :, None]
    first = pairs[..., 0:1]
    second = pairs[..., 1:2]
    rotated = torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)
    return torch.cat([leading, rotated.flatten(start_dim=-2)], dim=-1)


class HashRouter(nn.Module):
    def __init__(self, config: MiniConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_routed_experts
        self.route_scale = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        table = torch.empty(config.vocab_size, self.top_k, dtype=torch.long)
        for token_id in range(config.vocab_size):
            start = token_id * 5 + layer_idx * 3
            table[token_id] = torch.tensor(
                [(start + slot * 7) % self.num_experts for slot in range(self.top_k)],
                dtype=torch.long,
            )
        self.register_buffer("tid2eid", table, persistent=True)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_size)
        logits = F.linear(flat, self.weight)
        scores = sqrtsoftplus(logits)
        indices = self.tid2eid[input_ids.reshape(-1)]
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.route_scale, indices


class LearnedRouter(nn.Module):
    def __init__(self, config: MiniConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_routed_experts
        self.route_scale = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.register_buffer("correction_bias", torch.zeros(self.num_experts), persistent=True)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_size)
        logits = F.linear(flat, self.weight)
        scores = sqrtsoftplus(logits)
        indices = torch.topk(scores + self.correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.route_scale, indices


class HyperConnection(nn.Module):
    def __init__(self, config: MiniConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.iterations = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.input_norm = UnweightedRMSNorm(config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.ones(3))
        nn.init.normal_(self.fn, mean=0.0, std=0.02)
        nn.init.zeros_(self.base)

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_streams.ndim != 4 or hidden_streams.shape[-2] != self.hc_mult:
            raise ValueError(
                f"hidden_streams must have shape [batch, seq, {self.hc_mult}, hidden], "
                f"got {tuple(hidden_streams.shape)}"
            )
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.eps
        post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.iterations - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)

        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class GroupedLinear(nn.Module):
    def __init__(self, groups: int, in_per_group: int, out_per_group: int) -> None:
        super().__init__()
        self.groups = groups
        self.in_per_group = in_per_group
        self.out_per_group = out_per_group
        self.weight = nn.Parameter(torch.empty(groups, out_per_group, in_per_group))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_per_group)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.groups, self.in_per_group):
            raise ValueError(
                f"grouped input must end in {(self.groups, self.in_per_group)}, got {tuple(x.shape[-2:])}"
            )
        return torch.einsum("...gi,goi->...go", x, self.weight)


@dataclass(slots=True)
class CompressorState:
    buffer_kv: torch.Tensor
    buffer_gate: torch.Tensor
    compressed: torch.Tensor
    previous_ca_kv: torch.Tensor | None = None
    previous_ca_gate: torch.Tensor | None = None


@dataclass(slots=True)
class AttentionState:
    raw_kv: torch.Tensor
    compressor: CompressorState | None
    indexer: CompressorState | None
    next_position: int = 0


@dataclass(frozen=True, slots=True)
class AttentionTrace:
    raw_cache_length: int
    compressed_count: int
    indexer_compressed_count: int
    index_indices: tuple[int, ...]


class MiniCompressor(nn.Module):
    def __init__(
        self,
        config: MiniConfig,
        rate: int,
        output_dim: int,
        overlap: bool,
        rope_dim: int,
    ) -> None:
        super().__init__()
        self.rate = rate
        self.output_dim = output_dim
        self.overlap = overlap
        self.rope_dim = min(rope_dim, output_dim)
        projection_dim = output_dim * (2 if overlap else 1)
        self.kv_proj = nn.Linear(config.hidden_size, projection_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, projection_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(rate, projection_dim))
        self.norm = RMSNorm(output_dim, config.rms_norm_eps)
        self.theta = config.compress_rope_theta

    def new_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CompressorState:
        projection_dim = self.output_dim * (2 if self.overlap else 1)
        return CompressorState(
            buffer_kv=torch.empty(batch_size, 0, projection_dim, device=device, dtype=dtype),
            buffer_gate=torch.empty(batch_size, 0, projection_dim, device=device, dtype=dtype),
            compressed=torch.empty(batch_size, 0, self.output_dim, device=device, dtype=dtype),
        )

    def push(self, hidden_states: torch.Tensor, state: CompressorState) -> torch.Tensor:
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        state.buffer_kv = torch.cat([state.buffer_kv, kv], dim=1)
        state.buffer_gate = torch.cat([state.buffer_gate, gate], dim=1)
        if state.buffer_kv.shape[1] < self.rate:
            return state.compressed
        if state.buffer_kv.shape[1] != self.rate:
            raise RuntimeError("compressor step received more than one complete window")

        chunk_kv = state.buffer_kv
        chunk_gate = state.buffer_gate + self.position_bias.unsqueeze(0)
        if self.overlap:
            current_ca_kv = chunk_kv[..., : self.output_dim]
            current_ca_gate = chunk_gate[..., : self.output_dim]
            current_cb_kv = chunk_kv[..., self.output_dim :]
            current_cb_gate = chunk_gate[..., self.output_dim :]
            if state.previous_ca_kv is None:
                previous_kv = torch.zeros_like(current_ca_kv)
                previous_gate = torch.full_like(current_ca_gate, float("-inf"))
            else:
                previous_kv = state.previous_ca_kv
                previous_gate = state.previous_ca_gate
            pooled_kv = torch.cat([previous_kv, current_cb_kv], dim=1)
            pooled_gate = torch.cat([previous_gate, current_cb_gate], dim=1)
            state.previous_ca_kv = current_ca_kv.detach().clone()
            state.previous_ca_gate = current_ca_gate.detach().clone()
        else:
            pooled_kv = chunk_kv
            pooled_gate = chunk_gate

        weights = pooled_gate.softmax(dim=1, dtype=torch.float32).to(pooled_kv.dtype)
        compressed = self.norm((pooled_kv * weights).sum(dim=1))
        entry_position = state.compressed.shape[1] * self.rate
        positions = torch.full(
            (hidden_states.shape[0], 1),
            entry_position,
            device=hidden_states.device,
            dtype=torch.long,
        )
        cos, sin = _rope_cos_sin(positions, self.rope_dim, self.theta, compressed.dtype)
        compressed = _apply_partial_rope(compressed[:, None, None, :], cos, sin, self.rope_dim)[:, 0, 0]
        state.compressed = torch.cat([state.compressed, compressed.unsqueeze(1)], dim=1)
        state.buffer_kv = state.buffer_kv[:, :0]
        state.buffer_gate = state.buffer_gate[:, :0]
        return state.compressed


class MiniIndexer(nn.Module):
    def __init__(self, config: MiniConfig, rate: int) -> None:
        super().__init__()
        self.num_heads = config.index_num_heads
        self.head_dim = config.index_head_dim
        self.top_k = config.index_topk
        self.q_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weight_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.compressor = MiniCompressor(
            config,
            rate=rate,
            output_dim=self.head_dim,
            overlap=True,
            rope_dim=min(config.rope_head_dim, self.head_dim),
        )
        self.theta = config.compress_rope_theta
        self.rope_dim = min(config.rope_head_dim, self.head_dim)

    def new_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CompressorState:
        return self.compressor.new_state(batch_size, device, dtype)

    def step(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position: int,
        state: CompressorState,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        compressed = self.compressor.push(hidden_states, state)
        if compressed.shape[1] == 0:
            return compressed, ()
        batch = hidden_states.shape[0]
        q = self.q_proj(q_residual).view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        positions = torch.full((batch, 1), position, device=hidden_states.device, dtype=torch.long)
        cos, sin = _rope_cos_sin(positions, self.rope_dim, self.theta, q.dtype)
        q = _apply_partial_rope(q, cos, sin, self.rope_dim).transpose(1, 2)
        scores = torch.einsum("bshd,btd->bsht", q.float(), compressed.float())
        scores = F.relu(scores) * (self.head_dim**-0.5)
        head_weights = self.weight_proj(hidden_states).float() * (self.num_heads**-0.5)
        scores = (scores * head_weights.unsqueeze(-1)).sum(dim=2).squeeze(1)
        top_k = min(self.top_k, compressed.shape[1])
        indices = torch.topk(scores, top_k, dim=-1, sorted=True).indices
        if batch != 1:
            raise ValueError("the diagnostic mini-oracle currently records index traces for batch size one")
        return compressed, tuple(int(value) for value in indices[0].tolist())


class MiniAttention(nn.Module):
    def __init__(self, config: MiniConfig, layer_idx: int) -> None:
        super().__init__()
        if not 0 <= layer_idx < config.num_hidden_layers + config.num_mtp_layers:
            raise ValueError(f"layer_idx out of range: {layer_idx}")
        self.config = config
        self.layer_idx = layer_idx
        self.rate = config.compress_rates[layer_idx]
        self.layer_type = {
            0: "sliding_attention",
            4: "compressed_sparse_attention",
            128: "heavily_compressed_attention",
        }[self.rate]
        self.theta = config.rope_theta if self.rate == 0 else config.compress_rope_theta
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.rope_head_dim
        self.scaling = self.head_dim**-0.5
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_b_norm = UnweightedRMSNorm(config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        in_per_group = self.num_heads * self.head_dim // config.o_groups
        self.o_a_proj = GroupedLinear(config.o_groups, in_per_group, config.o_lora_rank)
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        if self.rate == 4:
            self.compressor: MiniCompressor | None = MiniCompressor(
                config,
                rate=4,
                output_dim=self.head_dim,
                overlap=True,
                rope_dim=self.rope_dim,
            )
            self.indexer: MiniIndexer | None = MiniIndexer(config, rate=4)
        elif self.rate == 128:
            self.compressor = MiniCompressor(
                config,
                rate=128,
                output_dim=self.head_dim,
                overlap=False,
                rope_dim=self.rope_dim,
            )
            self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

    def new_state(
        self,
        batch_size: int,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> AttentionState:
        device = torch.device(device)
        dtype = dtype or self.q_a_proj.weight.dtype
        compressor_state = (
            None if self.compressor is None else self.compressor.new_state(batch_size, device, dtype)
        )
        indexer_state = None if self.indexer is None else self.indexer.new_state(batch_size, device, dtype)
        return AttentionState(
            raw_kv=torch.empty(batch_size, 1, 0, self.head_dim, device=device, dtype=dtype),
            compressor=compressor_state,
            indexer=indexer_state,
        )

    def _attention_with_sink(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q.float(), kv.transpose(-1, -2).float()) * self.scaling
        sink = self.sinks.float().view(1, self.num_heads, 1, 1)
        maximum = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
        exp_scores = torch.exp(scores - maximum)
        sink_mass = torch.exp(sink - maximum)
        weights = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + sink_mass)
        return torch.matmul(weights.to(kv.dtype), kv)

    def step(
        self,
        hidden_states: torch.Tensor,
        position: int,
        state: AttentionState,
    ) -> tuple[torch.Tensor, AttentionTrace]:
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError(f"step expects [batch, 1, hidden], got {tuple(hidden_states.shape)}")
        if position != state.next_position:
            raise ValueError(f"expected position {state.next_position}, got {position}")
        batch = hidden_states.shape[0]
        positions = torch.full((batch, 1), position, device=hidden_states.device, dtype=torch.long)
        cos, sin = _rope_cos_sin(positions, self.rope_dim, self.theta, hidden_states.dtype)

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_b_norm(q)
        q = _apply_partial_rope(q, cos, sin, self.rope_dim)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(batch, 1, 1, self.head_dim)
        kv = _apply_partial_rope(kv, cos, sin, self.rope_dim)
        state.raw_kv = torch.cat([state.raw_kv, kv], dim=2)[:, :, -self.config.sliding_window :]

        compressed = hidden_states.new_empty(batch, 0, self.head_dim)
        if self.compressor is not None:
            if state.compressor is None:
                raise RuntimeError("missing compressor state")
            compressed = self.compressor.push(hidden_states, state.compressor)

        index_indices: tuple[int, ...] = ()
        indexer_count = 0
        if self.indexer is not None:
            if state.indexer is None:
                raise RuntimeError("missing indexer state")
            _, index_indices = self.indexer.step(hidden_states, q_residual, position, state.indexer)
            indexer_count = state.indexer.compressed.shape[1]

        compressed_for_attention = compressed
        if self.indexer is not None and compressed.shape[1] > 0:
            if index_indices:
                selected = torch.tensor(index_indices, device=compressed.device, dtype=torch.long)
                compressed_for_attention = compressed.index_select(1, selected)
            else:
                compressed_for_attention = compressed[:, :0]
        if compressed_for_attention.shape[1] > 0:
            kv_all = torch.cat([state.raw_kv, compressed_for_attention.unsqueeze(1)], dim=2)
        else:
            kv_all = state.raw_kv

        output = self._attention_with_sink(q, kv_all)
        output = _apply_partial_rope(output, cos, -sin, self.rope_dim)
        output = output.transpose(1, 2).reshape(batch, 1, self.config.o_groups, -1)
        output = self.o_a_proj(output).flatten(start_dim=2)
        output = self.o_b_proj(output)
        state.next_position += 1
        trace = AttentionTrace(
            raw_cache_length=state.raw_kv.shape[2],
            compressed_count=compressed.shape[1],
            indexer_compressed_count=indexer_count,
            index_indices=index_indices,
        )
        return output, trace

    def forward_full(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, list[AttentionTrace]]:
        if hidden_states.ndim != 3:
            raise ValueError(f"forward_full expects [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        state = self.new_state(hidden_states.shape[0], hidden_states.device, hidden_states.dtype)
        outputs: list[torch.Tensor] = []
        traces: list[AttentionTrace] = []
        for position in range(hidden_states.shape[1]):
            output, trace = self.step(hidden_states[:, position : position + 1], position, state)
            outputs.append(output)
            traces.append(trace)
        return torch.cat(outputs, dim=1), traces



@dataclass(frozen=True, slots=True)
class LayerStepTrace:
    attention: AttentionTrace
    route_indices: tuple[int, ...]
    route_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ModelStepTrace:
    layers: tuple[LayerStepTrace, ...]
    mtp: LayerStepTrace


@dataclass(slots=True)
class DecoderLayerState:
    attention: AttentionState


@dataclass(slots=True)
class ModelState:
    layers: list[DecoderLayerState]
    mtp: DecoderLayerState
    next_position: int = 0


@dataclass(frozen=True, slots=True)
class FullModelOutput:
    logits: torch.Tensor
    mtp_logits: torch.Tensor
    trace: tuple[ModelStepTrace, ...]


@dataclass(frozen=True, slots=True)
class StepModelOutput:
    logits: torch.Tensor
    mtp_logits: torch.Tensor
    trace: ModelStepTrace


class Expert(nn.Module):
    def __init__(self, config: MiniConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).float().clamp(max=self.limit)
        up = self.up_proj(hidden_states).float().clamp(min=-self.limit, max=self.limit)
        activated = F.silu(gate) * up
        return self.down_proj(activated.to(hidden_states.dtype))


class SparseMoE(nn.Module):
    def __init__(self, config: MiniConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx < config.num_hash_layers:
            self.router: HashRouter | LearnedRouter = HashRouter(config, layer_idx)
        else:
            self.router = LearnedRouter(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_routed_experts)])
        self.shared_expert = Expert(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden_size = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_size)
        if isinstance(self.router, HashRouter):
            _, route_weights, route_indices = self.router(hidden_states, input_ids)
        else:
            _, route_weights, route_indices = self.router(hidden_states)

        routed = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            hits = torch.nonzero(route_indices == expert_idx, as_tuple=False)
            if hits.numel() == 0:
                continue
            token_positions = hits[:, 0]
            route_slots = hits[:, 1]
            contribution = expert(flat.index_select(0, token_positions))
            contribution = contribution * route_weights[token_positions, route_slots].unsqueeze(-1).to(
                contribution.dtype
            )
            routed.index_add_(0, token_positions, contribution.to(routed.dtype))
        shared = self.shared_expert(flat)
        output = (routed + shared).reshape(batch, seq_len, hidden_size)
        return (
            output,
            route_indices.reshape(batch, seq_len, -1),
            route_weights.reshape(batch, seq_len, -1),
        )


class HyperHead(nn.Module):
    def __init__(self, config: MiniConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.eps = config.hc_eps
        self.input_norm = UnweightedRMSNorm(config.rms_norm_eps)
        self.fn = nn.Parameter(torch.empty(config.hc_mult, config.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.zeros(config.hc_mult))
        self.scale = nn.Parameter(torch.ones(1))
        nn.init.normal_(self.fn, mean=0.0, std=0.02)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(streams.flatten(start_dim=2).float())
        mixes = F.linear(flat, self.fn.float())
        pre = torch.sigmoid(mixes * self.scale.float() + self.base.float()) + self.eps
        return (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, config: MiniConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention = MiniAttention(config, layer_idx)
        self.moe = SparseMoE(config, layer_idx)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def new_state(
        self,
        batch_size: int,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> DecoderLayerState:
        return DecoderLayerState(self.attention.new_state(batch_size, device, dtype))

    @staticmethod
    def _mix(
        streams: torch.Tensor,
        sublayer_output: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        dtype = streams.dtype
        residual = torch.matmul(comb.to(dtype).transpose(-1, -2), streams)
        injected = post.to(dtype).unsqueeze(-1) * sublayer_output.unsqueeze(-2)
        return residual + injected

    @staticmethod
    def _trace_for_position(
        attention_trace: AttentionTrace,
        route_indices: torch.Tensor,
        route_weights: torch.Tensor,
        position: int,
    ) -> LayerStepTrace:
        if route_indices.shape[0] != 1:
            raise ValueError("the diagnostic trace format currently supports batch size one")
        return LayerStepTrace(
            attention=attention_trace,
            route_indices=tuple(int(value) for value in route_indices[0, position].tolist()),
            route_weights=tuple(float(value) for value in route_weights[0, position].float().tolist()),
        )

    def forward_full(
        self,
        streams: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[LayerStepTrace]]:
        post, comb, collapsed = self.attention_hc(streams)
        attention_output, attention_traces = self.attention.forward_full(self.input_norm(collapsed))
        streams = self._mix(streams, attention_output, post, comb)

        post, comb, collapsed = self.ffn_hc(streams)
        moe_output, route_indices, route_weights = self.moe(self.post_attention_norm(collapsed), input_ids)
        streams = self._mix(streams, moe_output, post, comb)
        traces = [
            self._trace_for_position(attention_traces[position], route_indices, route_weights, position)
            for position in range(input_ids.shape[1])
        ]
        return streams, traces

    def step(
        self,
        streams: torch.Tensor,
        input_ids: torch.Tensor,
        position: int,
        state: DecoderLayerState,
    ) -> tuple[torch.Tensor, LayerStepTrace]:
        post, comb, collapsed = self.attention_hc(streams)
        attention_output, attention_trace = self.attention.step(
            self.input_norm(collapsed), position, state.attention
        )
        streams = self._mix(streams, attention_output, post, comb)

        post, comb, collapsed = self.ffn_hc(streams)
        moe_output, route_indices, route_weights = self.moe(self.post_attention_norm(collapsed), input_ids)
        streams = self._mix(streams, moe_output, post, comb)
        trace = self._trace_for_position(attention_trace, route_indices, route_weights, 0)
        return streams, trace


class MTPBlock(nn.Module):
    def __init__(
        self,
        config: MiniConfig,
        embed_tokens: nn.Embedding,
        head: nn.Linear,
    ) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.head = head
        self.embedding_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.embedding_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.hidden_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.decoder = DecoderLayer(config, config.num_hidden_layers)
        self.hc_head = HyperHead(config)
        self.output_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def new_state(
        self,
        batch_size: int,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> DecoderLayerState:
        return self.decoder.new_state(batch_size, device, dtype)

    def _inputs(self, streams: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_proj(self.embedding_norm(self.embed_tokens(input_ids))).unsqueeze(2)
        hidden = self.hidden_proj(self.hidden_norm(streams))
        return embedding + hidden

    def forward_full(
        self,
        streams: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[LayerStepTrace]]:
        streams, traces = self.decoder.forward_full(self._inputs(streams, input_ids), input_ids)
        logits = self.head(self.output_norm(self.hc_head(streams)))
        return logits, traces

    def step(
        self,
        streams: torch.Tensor,
        input_ids: torch.Tensor,
        position: int,
        state: DecoderLayerState,
    ) -> tuple[torch.Tensor, LayerStepTrace]:
        streams, trace = self.decoder.step(self._inputs(streams, input_ids), input_ids, position, state)
        logits = self.head(self.output_norm(self.hc_head(streams)))
        return logits, trace


class MiniReferenceModel(nn.Module):
    def __init__(self, config: MiniConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.hc_head = HyperHead(config)
        self.output_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = MTPBlock(config, self.embed_tokens, self.lm_head)

    def new_state(
        self,
        batch_size: int,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> ModelState:
        return ModelState(
            layers=[layer.new_state(batch_size, device, dtype) for layer in self.layers],
            mtp=self.mtp.new_state(batch_size, device, dtype),
        )

    def _expand_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embed_tokens(input_ids)
        return embedded.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

    def forward_full(self, input_ids: torch.Tensor) -> FullModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        streams = self._expand_embeddings(input_ids)
        traces_by_layer: list[list[LayerStepTrace]] = []
        for layer in self.layers:
            streams, traces = layer.forward_full(streams, input_ids)
            traces_by_layer.append(traces)
        logits = self.lm_head(self.output_norm(self.hc_head(streams)))
        mtp_logits, mtp_traces = self.mtp.forward_full(streams, input_ids)
        steps = tuple(
            ModelStepTrace(
                layers=tuple(traces_by_layer[layer_idx][position] for layer_idx in range(len(self.layers))),
                mtp=mtp_traces[position],
            )
            for position in range(input_ids.shape[1])
        )
        return FullModelOutput(logits=logits, mtp_logits=mtp_logits, trace=steps)

    def step(self, input_ids: torch.Tensor, state: ModelState) -> StepModelOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError(f"step input_ids must be [batch, 1], got {tuple(input_ids.shape)}")
        if state.next_position >= self.config.max_seq_len:
            raise ValueError("decode state reached max_seq_len")
        position = state.next_position
        streams = self._expand_embeddings(input_ids)
        layer_traces: list[LayerStepTrace] = []
        for layer, layer_state in zip(self.layers, state.layers, strict=True):
            streams, trace = layer.step(streams, input_ids, position, layer_state)
            layer_traces.append(trace)
        logits = self.lm_head(self.output_norm(self.hc_head(streams)))
        mtp_logits, mtp_trace = self.mtp.step(streams, input_ids, position, state.mtp)
        state.next_position += 1
        return StepModelOutput(
            logits=logits,
            mtp_logits=mtp_logits,
            trace=ModelStepTrace(layers=tuple(layer_traces), mtp=mtp_trace),
        )

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if input_ids.shape[1] + max_new_tokens > self.config.max_seq_len:
            raise ValueError("generation would exceed max_seq_len")
        generated = input_ids.clone()
        state = self.new_state(input_ids.shape[0], input_ids.device, self.embed_tokens.weight.dtype)
        last_output: StepModelOutput | None = None
        for position in range(input_ids.shape[1]):
            last_output = self.step(input_ids[:, position : position + 1], state)
        if last_output is None:
            raise ValueError("generation requires at least one prompt token")
        for _ in range(max_new_tokens):
            next_token = last_output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            last_output = self.step(next_token, state)
        return generated
