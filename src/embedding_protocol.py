"""Model-specific text-embedding protocol, independent from the RL objective.

An embedding checkpoint is more than transformer weights: its pooling rule, padding
side, terminal token, and text templates are part of the learned model.  This module
keeps those choices in one small adapter layer so GRPO only consumes unit vectors.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor


SUPPORTED_POOLING_METHODS = ("last", "mean", "cls")
SUPPORTED_APPEND_TOKENS = ("none", "eos", "pad")
SUPPORTED_PADDING_SIDES = ("left", "right")
EMBEDDING_PROTOCOL_FILENAME = "embedding_protocol.json"


def protocol_from_model_args(model_args) -> dict[str, object]:
    return {
        "pooling_method": model_args.pooling_method,
        "padding_side": model_args.padding_side,
        "append_token": model_args.append_token,
        "query_prompt_template": model_args.query_prompt_template,
        "document_prompt_template": model_args.document_prompt_template,
        "max_length": model_args.embedding_max_length,
        "normalize": True,
    }


def save_embedding_protocol(model_args, output_dir: str | Path) -> None:
    output_path = Path(output_dir) / EMBEDDING_PROTOCOL_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(protocol_from_model_args(model_args), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_embedding_protocol(model_path: str | Path) -> dict[str, object]:
    protocol_path = Path(model_path) / EMBEDDING_PROTOCOL_FILENAME
    if not protocol_path.is_file():
        return {}
    with protocol_path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if not isinstance(protocol, dict):
        raise ValueError(f"Expected a JSON object in {protocol_path}")
    return protocol


def protocol_to_eval_kwargs(protocol: dict[str, object]) -> dict[str, object]:
    if not protocol:
        return {}
    return {
        "pooler_type": protocol["pooling_method"],
        "padding_side": protocol["padding_side"],
        "append_token": protocol["append_token"],
        "query_prompt_template": protocol["query_prompt_template"],
        "document_prompt_template": protocol["document_prompt_template"],
        "max_length": protocol.get("max_length", 8192),
        "do_norm": protocol.get("normalize", True),
        "use_instruction": "{task_description}" in str(protocol["query_prompt_template"]),
    }


def validate_embedding_protocol(
    *,
    pooling_method: str,
    padding_side: str,
    append_token: str,
    query_prompt_template: str,
    document_prompt_template: str,
) -> None:
    if pooling_method not in SUPPORTED_POOLING_METHODS:
        raise ValueError(
            f"Unsupported pooling_method: {pooling_method!r}. "
            f"Expected one of {SUPPORTED_POOLING_METHODS}."
        )
    if padding_side not in SUPPORTED_PADDING_SIDES:
        raise ValueError(
            f"Unsupported padding_side: {padding_side!r}. "
            f"Expected one of {SUPPORTED_PADDING_SIDES}."
        )
    if append_token not in SUPPORTED_APPEND_TOKENS:
        raise ValueError(
            f"Unsupported append_token: {append_token!r}. "
            f"Expected one of {SUPPORTED_APPEND_TOKENS}."
        )
    for name, template in (
        ("query_prompt_template", query_prompt_template),
        ("document_prompt_template", document_prompt_template),
    ):
        if not isinstance(template, str) or not template:
            raise ValueError(f"{name} must be a non-empty string")
        if not any(field in template for field in ("{text}", "{query}", "{document}")):
            raise ValueError(
                f"{name} must contain one of {{text}}, {{query}}, or {{document}}; "
                f"got {template!r}"
            )


def format_embedding_text(
    template: str,
    text: str,
    *,
    task_description: str = "",
) -> str:
    """Format either role template while accepting explicit or generic placeholders."""
    try:
        return template.format(
            text=text,
            query=text,
            document=text,
            task_description=task_description,
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown placeholder {exc.args[0]!r} in embedding prompt template {template!r}"
        ) from exc


def append_configured_token(
    texts: Sequence[str],
    tokenizer,
    append_token: str,
) -> list[str]:
    """Append the configured tokenizer token as text before batch tokenization."""
    if append_token == "none":
        return list(texts)
    token = getattr(tokenizer, f"{append_token}_token", None)
    if not token:
        raise ValueError(
            f"append_token={append_token!r} requires tokenizer.{append_token}_token to be set"
        )
    return [text + token for text in texts]


def pool_embeddings(
    last_hidden_states: Tensor,
    attention_mask: Tensor,
    *,
    pooling_method: str,
    normalize: bool = True,
) -> Tensor:
    """Pool transformer hidden states according to a checkpoint's native protocol."""
    if last_hidden_states.ndim != 3:
        raise ValueError(
            "last_hidden_states must have shape [batch, sequence, hidden], "
            f"got {tuple(last_hidden_states.shape)}"
        )
    if attention_mask.ndim != 2 or attention_mask.shape != last_hidden_states.shape[:2]:
        raise ValueError(
            "attention_mask must match the first two hidden-state dimensions, "
            f"got mask={tuple(attention_mask.shape)} hidden={tuple(last_hidden_states.shape)}"
        )

    if pooling_method == "last":
        # Works for both left- and right-padded batches.  With left padding every
        # sequence ends at the final position; otherwise gather its final real token.
        if bool(torch.all(attention_mask[:, -1] == 1)):
            embeddings = last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
            embeddings = last_hidden_states[batch_indices, sequence_lengths]
    elif pooling_method == "mean":
        mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
        embeddings = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    elif pooling_method == "cls":
        # Encoder checkpoints conventionally use right padding, leaving CLS at index 0.
        embeddings = last_hidden_states[:, 0]
    else:
        raise ValueError(
            f"Unsupported pooling_method: {pooling_method!r}. "
            f"Expected one of {SUPPORTED_POOLING_METHODS}."
        )

    return F.normalize(embeddings, dim=-1, p=2) if normalize else embeddings
