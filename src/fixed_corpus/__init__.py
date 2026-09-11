"""Frozen-corpus infrastructure shared by fixed-candidate and RAG training."""

from .index import (
    FrozenCorpusIndex,
    FrozenCorpusIndexRouter,
    FrozenDistributedIndex,
    load_frozen_index,
    load_index_manifest,
    sha256_file,
    validate_frozen_protocol,
    write_frozen_training_audit,
)
__all__ = [
    "FixedCorpusGRPOModel",
    "FrozenCandidateScorer",
    "FrozenCorpusIndex",
    "FrozenCorpusIndexRouter",
    "FrozenDistributedIndex",
    "load_frozen_index",
    "QueryOnlyRLWrapper",
    "QueryOnlySupervisedModel",
    "QueryPolicyHead",
    "QueryPolicyOutput",
    "load_index_manifest",
    "sha256_file",
    "validate_frozen_protocol",
    "write_frozen_training_audit",
]


def __getattr__(name: str):
    """Keep the package API convenient without eagerly importing training stacks."""
    if name in {
        "FixedCorpusGRPOModel",
        "FrozenCandidateScorer",
        "QueryOnlySupervisedModel",
    }:
        from . import models

        return getattr(models, name)
    if name in {"QueryOnlyRLWrapper", "QueryPolicyHead", "QueryPolicyOutput"}:
        from . import policy

        return getattr(policy, name)
    raise AttributeError(name)
