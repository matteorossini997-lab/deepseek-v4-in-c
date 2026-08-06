from __future__ import annotations

import hashlib
import platform
from dataclasses import asdict
from typing import Any

import torch

from .config import MiniConfig
from .initialization import initialize_reference_model
from .reference import LayerStepTrace, ModelStepTrace


def _tensor_checksum(tensor: torch.Tensor) -> str:
    raw = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(torch.uint8).flatten().tolist()
    data = bytes(raw)
    return hashlib.sha256(data).hexdigest()


def _token_sequence(length: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.long)
    return ((positions * 17 + 3) % vocab_size).unsqueeze(0)


def _round_values(values: tuple[float, ...], digits: int = 9) -> list[float]:
    return [round(value, digits) for value in values]


def _layer_trace_to_dict(trace: LayerStepTrace) -> dict[str, Any]:
    return {
        "attention": asdict(trace.attention),
        "route_indices": list(trace.route_indices),
        "route_weights": _round_values(trace.route_weights),
    }


def _step_trace_to_dict(trace: ModelStepTrace) -> dict[str, Any]:
    return {
        "layers": [_layer_trace_to_dict(layer) for layer in trace.layers],
        "mtp": _layer_trace_to_dict(trace.mtp),
    }


def _routes_equal(first: tuple[ModelStepTrace, ...], second: list[ModelStepTrace]) -> bool:
    if len(first) != len(second):
        return False
    for left, right in zip(first, second, strict=True):
        if left.mtp.route_indices != right.mtp.route_indices:
            return False
        if left.mtp.attention.index_indices != right.mtp.attention.index_indices:
            return False
        for left_layer, right_layer in zip(left.layers, right.layers, strict=True):
            if left_layer.route_indices != right_layer.route_indices:
                return False
            if left_layer.attention.index_indices != right_layer.attention.index_indices:
                return False
            if left_layer.attention.compressed_count != right_layer.attention.compressed_count:
                return False
            if left_layer.attention.indexer_compressed_count != right_layer.attention.indexer_compressed_count:
                return False
    return True


def _selected_logits(logits: torch.Tensor, positions: list[int], width: int = 16) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for position in positions:
        row = logits[0, position].float()
        top_values, top_indices = torch.topk(row, min(8, row.numel()), sorted=True)
        result[str(position)] = {
            "prefix": [round(float(value), 8) for value in row[:width].tolist()],
            "argmax": int(row.argmax().item()),
            "top_indices": [int(value) for value in top_indices.tolist()],
            "top_values": [round(float(value), 8) for value in top_values.tolist()],
        }
    return result


def _incremental(model, input_ids: torch.Tensor):
    state = model.new_state(input_ids.shape[0], input_ids.device, model.embed_tokens.weight.dtype)
    logits: list[torch.Tensor] = []
    mtp_logits: list[torch.Tensor] = []
    traces: list[ModelStepTrace] = []
    for position in range(input_ids.shape[1]):
        output = model.step(input_ids[:, position : position + 1], state)
        logits.append(output.logits)
        mtp_logits.append(output.mtp_logits)
        traces.append(output.trace)
    return torch.cat(logits, dim=1), torch.cat(mtp_logits, dim=1), traces


def build_oracle(
    config: MiniConfig,
    device: str | torch.device = "cpu",
    sequence_length: int = 130,
) -> dict[str, Any]:
    if not 1 <= sequence_length <= config.max_seq_len:
        raise ValueError(f"sequence_length must be in [1, {config.max_seq_len}]")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")

    model = initialize_reference_model(config, device=torch_device).eval()
    input_ids = _token_sequence(sequence_length, config.vocab_size, torch_device)
    with torch.inference_mode():
        full = model.forward_full(input_ids)
        incremental_logits, incremental_mtp, incremental_traces = _incremental(model, input_ids)
        generation_prompt = input_ids[:, : min(7, sequence_length)]
        generated = model.generate(generation_prompt, max_new_tokens=min(5, config.max_seq_len - generation_prompt.shape[1]))

    main_error = (full.logits.float() - incremental_logits.float()).abs()
    mtp_error = (full.mtp_logits.float() - incremental_mtp.float()).abs()
    selected_positions = [
        position
        for position in (0, 1, 3, 4, 15, 16, 17, 126, 127, 128, 129)
        if position < sequence_length
    ]
    argmax_full = full.logits.argmax(dim=-1)
    argmax_incremental = incremental_logits.argmax(dim=-1)
    mtp_argmax_full = full.mtp_logits.argmax(dim=-1)
    mtp_argmax_incremental = incremental_mtp.argmax(dim=-1)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device_type": torch_device.type,
        "dtype": str(model.embed_tokens.weight.dtype).removeprefix("torch."),
    }
    if torch_device.type == "cuda":
        environment["device_name"] = torch.cuda.get_device_name(torch_device)
        environment["compute_capability"] = list(torch.cuda.get_device_capability(torch_device))

    return {
        "schema_version": 1,
        "model": {
            "architecture": "deepseek_v4_flash_mini",
            "parameter_count": parameter_count,
            "parameter_bytes_fp32": parameter_bytes,
            "config": config.to_dict(),
        },
        "environment": environment,
        "sequence_length": sequence_length,
        "selected_positions": selected_positions,
        "input_ids": [int(value) for value in input_ids[0].tolist()],
        "teacher_forced_argmax": [int(value) for value in argmax_full[0].tolist()],
        "incremental_argmax": [int(value) for value in argmax_incremental[0].tolist()],
        "mtp_teacher_forced_argmax": [int(value) for value in mtp_argmax_full[0].tolist()],
        "mtp_incremental_argmax": [int(value) for value in mtp_argmax_incremental[0].tolist()],
        "generated_ids": [int(value) for value in generated[0].tolist()],
        "parity": {
            "argmax_equal": bool(torch.equal(argmax_full, argmax_incremental)),
            "mtp_argmax_equal": bool(torch.equal(mtp_argmax_full, mtp_argmax_incremental)),
            "routes_equal": _routes_equal(full.trace, incremental_traces),
            "max_abs_error": float(max(main_error.max().item(), mtp_error.max().item())),
            "main_max_abs_error": float(main_error.max().item()),
            "mtp_max_abs_error": float(mtp_error.max().item()),
        },
        "checksums": {
            "full_logits": _tensor_checksum(full.logits),
            "incremental_logits": _tensor_checksum(incremental_logits),
            "full_mtp_logits": _tensor_checksum(full.mtp_logits),
            "incremental_mtp_logits": _tensor_checksum(incremental_mtp),
        },
        "selected_logits": _selected_logits(full.logits, selected_positions),
        "selected_mtp_logits": _selected_logits(full.mtp_logits, selected_positions),
        "trace": {str(position): _step_trace_to_dict(full.trace[position]) for position in selected_positions},
    }
