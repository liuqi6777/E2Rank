"""E0 anchor vectors for constraining query-encoder drift.

Round 3 showed why the best arm still fails to beat its initialization by much:
`RL-GradedNDCG` gains +0.0214 `answer_mrr@10` in the training domain and gives
back -0.0072 on the five held-out datasets. Training moves the query embedding
off the manifold the frozen document index was built for; the gain it buys
in-domain costs generalization everywhere else.

An anchor penalty bounds that drift directly: each training query carries the
frozen E0 embedding it had at initialization, and the loss adds
``rag_anchor_coef * (1 - cos(e_theta(q), e_E0(q)))``. At coefficient 0 this is
the round-3 recipe exactly; as the coefficient grows the encoder is pinned to
E0 and the update shrinks toward nothing. It is the knob between those ends
that round 4 sweeps.

The anchors are precomputed once with the *initial* weights, before any
optimizer step, and cached as a single fp16 memmap alongside a metadata JSON
that pins the manifest hash, model revision, split and row count. A cache that
does not match the current dataset is ignored and rebuilt rather than silently
reused -- a stale anchor row would regularize the wrong query.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainerCallback

from embedding_protocol import format_embedding_text, pool_embeddings
from rag.data import RAG_TASK_DESCRIPTION


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def anchor_cache_path(manifest_path: str | Path, cache_dir: str | Path, variant: str = "") -> Path:
    stem = Path(manifest_path).stem
    suffix = ""
    if variant:
        digest = hashlib.sha256(variant.encode("utf-8")).hexdigest()[:10]
        suffix = f".{digest}"
    return Path(cache_dir) / f"{stem}{suffix}.anchors.fp16.npy"


def _meta_path(vector_path: Path) -> Path:
    return vector_path.with_suffix(vector_path.suffix + ".meta.json")


def validate_anchor_cache(
    vector_path: str | Path,
    *,
    manifest_path: str | Path,
    model_name_or_path: str,
    model_revision: str | None,
    row_count: int,
    dimension: int,
) -> bool:
    """True when the cached anchors match the current dataset and encoder."""
    vector_path = Path(vector_path)
    meta_path = _meta_path(vector_path)
    if not vector_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (ValueError, OSError):
        return False
    expected = {
        "manifest_sha256": _sha256_file(manifest_path),
        "model_name_or_path": model_name_or_path,
        "model_revision": model_revision,
        "row_count": int(row_count),
        "dimension": int(dimension),
    }
    return all(meta.get(key) == value for key, value in expected.items())


def write_anchor_meta(
    vector_path: str | Path,
    *,
    manifest_path: str | Path,
    model_name_or_path: str,
    model_revision: str | None,
    row_count: int,
    dimension: int,
) -> None:
    meta = {
        "manifest_sha256": _sha256_file(manifest_path),
        "model_name_or_path": model_name_or_path,
        "model_revision": model_revision,
        "row_count": int(row_count),
        "dimension": int(dimension),
        "format": "fp16 memmap, row i is the anchor of dataset row i",
    }
    _meta_path(Path(vector_path)).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


@torch.no_grad()
def encode_anchor_rows(
    *,
    model,
    tokenizer,
    dataset,
    row_indices: list[int],
    vector_path: str | Path,
    dimension: int,
    batch_size: int,
    device: torch.device,
    query_prompt_template: str,
    query_max_length: int = 128,
    append_token: str = "pad",
) -> None:
    """Encode ``row_indices`` of ``dataset`` with the current (initial) weights.

    Each rank writes a disjoint set of rows of one shared fp16 memmap, so no
    merge step is needed; the caller synchronizes with a barrier afterwards.
    Must run before any optimizer step: the anchors are the *initial* encoder's
    view of each query. ``dimension`` is the model's hidden size, known before
    encoding starts, so the memmap can be opened up front.
    """
    was_training = model.training
    model.eval()
    vector_path = Path(vector_path)
    from embedding_protocol import tokenize_embedding_texts

    # np.memmap's "w+" truncates on every open, which would let each rank wipe
    # the rows the other ranks have already written. Pre-size the file once and
    # then open it "r+": writes stay confined to the caller's own rows.
    expected_bytes = len(dataset) * dimension * np.dtype(np.float16).itemsize
    if not vector_path.is_file() or vector_path.stat().st_size != expected_bytes:
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vector_path, "wb") as handle:
            handle.truncate(expected_bytes)
    sink = np.memmap(
        vector_path, dtype=np.float16, mode="r+", shape=(len(dataset), dimension)
    )
    try:
        for start in range(0, len(row_indices), batch_size):
            chunk = row_indices[start : start + batch_size]
            records = [dataset[row] for row in chunk]
            texts = [
                format_embedding_text(
                    query_prompt_template, record["question"], task_description=RAG_TASK_DESCRIPTION
                )
                for record in records
            ]
            tokenized = tokenize_embedding_texts(
                texts, tokenizer, append_token, max_length=query_max_length
            )
            inputs = {key: value.to(device) for key, value in dict(tokenized).items()}
            # The Trainer hands this function its full model -- under DeepSpeed
            # that is the engine wrapping RAGRLModel, whose forward() takes
            # (query, reward_inputs) for the RL loss and rejects a raw token
            # batch (trial 2088952). encode_query is the wrappers' own query
            # path, pooling and normalization included, and is what the probe
            # and tuning callbacks already call; a bare backbone falls back to
            # a direct forward.
            if hasattr(model, "encode_query"):
                embeddings = model.encode_query(inputs)
            else:
                hidden = model(**inputs).last_hidden_state
                embeddings = pool_embeddings(
                    hidden, inputs["attention_mask"], pooling_method="last", normalize=True
                )
            embeddings = embeddings.float().cpu().numpy().astype(np.float16)
            for position, row in enumerate(chunk):
                sink[row] = embeddings[position]
            if (start // batch_size) % 8 == 0:
                sink.flush()
    finally:
        sink.flush()
        model.train(was_training)


def open_anchors(vector_path: str | Path, row_count: int, dimension: int) -> np.memmap:
    return np.memmap(
        Path(vector_path), dtype=np.float16, mode="r", shape=(row_count, dimension)
    )


def barrier_if_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class RAGAnchorPrecomputeCallback(TrainerCallback):
    """Build the anchor cache at ``on_train_begin``, through the wrapped model.

    The precompute cannot run earlier in ``train_rag.main``: the backbone is
    loaded under DeepSpeed's ZeRO-3 init context, so before the Trainer exists
    its parameters are partitioned and a plain forward raises ``'weight' must
    be 2-D`` (observed on trials 2081553/2081554). At ``on_train_begin`` the
    Trainer's model is the DeepSpeed engine, whose forward gathers on the fly
    -- and no optimizer step has run yet, so the weights are still the
    initialization the anchors are supposed to capture.

    Subclassing ``TrainerCallback`` is not cosmetic: the ``CallbackHandler``
    fires every event on every registered callback, starting with
    ``on_init_end`` inside ``Trainer.__init__`` -- before ``on_train_begin``
    ever runs. A plain class without the inherited no-op events dies there
    with ``AttributeError`` (observed on trials 2082629/2082630).
    """

    def __init__(
        self,
        *,
        dataset,
        tokenizer,
        training_args,
        data_args,
        model_args,
        dimension: int,
        vector_path,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        self.dimension = int(dimension)
        self.vector_path = Path(vector_path)

    def __call__(self, args, state, control, model=None, **kwargs):
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        world_size = dist.get_world_size() if distributed else 1
        row_count = len(self.dataset)
        if rank == 0:
            self.vector_path.unlink(missing_ok=True)
            write_anchor_meta(
                self.vector_path,
                manifest_path=self.data_args.rag_candidate_manifest,
                model_name_or_path=self.model_args.model_name_or_path,
                model_revision=self.model_args.model_revision,
                row_count=row_count,
                dimension=self.dimension,
            )
            # Pre-size before any rank opens it: np.memmap "w+" truncates on
            # open, so later opens must find the file at full size.
            self.vector_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.vector_path, "wb") as handle:
                handle.truncate(row_count * self.dimension * np.dtype(np.float16).itemsize)
        barrier_if_distributed()
        rows = list(range(rank, row_count, world_size))
        print(
            f"[rag-anchors] encoding {len(rows)}/{row_count} rows on rank {rank} "
            f"-> {self.vector_path}",
            flush=True,
        )
        encode_anchor_rows(
            model=model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            row_indices=rows,
            vector_path=self.vector_path,
            dimension=self.dimension,
            batch_size=self.training_args.per_device_train_batch_size,
            device=args.device if hasattr(args, "device") else torch.device("cuda"),
            query_prompt_template=self.model_args.query_prompt_template,
            query_max_length=min(
                self.data_args.rag_query_max_length, self.model_args.embedding_max_length
            ),
            append_token=self.model_args.append_token,
        )
        barrier_if_distributed()
        print(f"[rag-anchors] ready: {self.vector_path}", flush=True)
        self.dataset.attach_anchors(str(self.vector_path), self.dimension)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self(args, state, control, model=model, **kwargs)
