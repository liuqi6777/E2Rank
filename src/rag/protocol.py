from __future__ import annotations


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
    expected_model = index_manifest.get("model_name_or_path")
    if expected_model and expected_model != model_name_or_path:
        raise ValueError(
            f"Query model {model_name_or_path!r} differs from indexed document model {expected_model!r}"
        )
    expected_revision = index_manifest.get("resolved_model_revision")
    if expected_revision and resolved_model_revision and expected_revision != resolved_model_revision:
        raise ValueError("Query model revision differs from the frozen document encoder revision")
    for key, actual in (
        ("pooling_method", pooling_method),
        ("padding_side", padding_side),
        ("append_token", append_token),
    ):
        expected = index_manifest.get(key)
        if expected is not None and expected != actual:
            raise ValueError(f"Query/index protocol mismatch for {key}: {actual!r} != {expected!r}")
