"""Backward-compatible imports for the shared frozen-corpus store."""

from fixed_corpus.index import (
    FrozenCorpusIndex,
    FrozenDistributedIndex,
    load_index_manifest,
    sha256_file,
    validate_frozen_protocol,
)

__all__ = [
    "FrozenCorpusIndex",
    "FrozenDistributedIndex",
    "load_index_manifest",
    "sha256_file",
    "validate_frozen_protocol",
]
