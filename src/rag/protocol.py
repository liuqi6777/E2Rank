from __future__ import annotations

from frozen_corpus import validate_frozen_protocol


def validate_query_index_protocol(
    index_manifest: dict,
    *,
    model_name_or_path: str,
    resolved_model_revision: str | None,
    pooling_method: str,
    padding_side: str,
    append_token: str,
) -> None:
    """Reject query/index embedding-space drift before retrieval starts."""
    validate_frozen_protocol(
        index_manifest,
        model_name_or_path=model_name_or_path,
        resolved_model_revision=resolved_model_revision,
        pooling_method=pooling_method,
        padding_side=padding_side,
        append_token=append_token,
    )
