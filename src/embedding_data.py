import glob
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterator, Sequence

import torch
import transformers
from torch.utils.data import Dataset, Sampler

from embedding_protocol import append_configured_token, format_embedding_text


TASK_PROMPTS = {
    "msmarco": "Given a web search query, retrieve the documents that answer the query",
    "nq": "Given a question, retrieve Wikipedia documents that answer the question",
    "hotpotqa": "Given a multi-hop question, retrieve the documents that can help answer the question",
    "trivia": "Retrieve Wikipedia documents that answer the question",
    "t2ranking": "Given a Chinese search query, retrieve the documents that answer the query",
    "dureader": "Given a Chinese search query, retrieve the documents that answer the query",
    "mmarco_chinese": "Given a Chinese web search query, retrieve the documents that answer the query",
    "cMedQAv2": "Given a Chinese medical question, retrieve the documents that answer the question",
    "miracl": "Given a question, retrieve Wikipedia documents that answer the question",
    "allnli": "Given a premise, retrieve a hypothesis that is entailed by the premise",
    "fever": "Given a claim, retrieve documents that support or refute the claim",
    "eli5_question_answer": "Given a question, retrieve the answer that explains it",
    "squad": "Given a question, retrieve a Wikipedia passage that answers the question",
    "quora_duplicates": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "mrtydi": "Given a question, retrieve Wikipedia documents that answer the question",
    "mldr": "Given a query, retrieve the long documents that are relevant to the query",
    "law_medical": "Given a Chinese question, retrieve legal or medical documents that answer the question",
    "zh_nli": "Given a premise, retrieve a hypothesis that is entailed by the premise",
}

DEFAULT_TASK_PROMPTS = "Given a query, retrieve the documents that are relevant to the query"


# Maps a BGE-M3 source subdirectory name to a TASK_PROMPTS key. Directory names whose
# lowercased form already matches a TASK_PROMPTS key (e.g. "cMedQAv2") need no entry.
SOURCE_DIR_TO_TASK = {
    "msmarco": "msmarco",
    "nq": "nq",
    "hotpotqa": "hotpotqa",
    "trivia": "trivia",
    "t2ranking": "t2ranking",
    "dureader": "dureader",
    "mmarco-zh": "mmarco_chinese",
    "cmedqav2": "cMedQAv2",
    "miracl": "miracl",
    "en_nli_data": "allnli",
    "zh_nli_data": "zh_nli",
    "squad": "squad",
    "mr.tydi": "mrtydi",
    "mrtydi": "mrtydi",
    "mldr": "mldr",
    "law-medical_data": "law_medical",
}


def _source_name_from_dir(dir_name: str) -> str:
    """Resolve a source subdirectory name to a TASK_PROMPTS key.

    Falls back to the lowercased directory name (which then hits DEFAULT_TASK_PROMPTS
    in ``_format_query`` if it is not a registered task).
    """
    key = dir_name.strip().lower()
    return SOURCE_DIR_TO_TASK.get(key, key)


def _length_bucket_from_path(path: str) -> str:
    """Extract the ``len-<lo>-<hi>`` length-bucket tag from a data filename.

    Files are named like ``dureader_len-0-500.jsonl``; the tag is what distinguishes
    one length bucket from another within the same source. Returns the substring from
    ``len-`` to the extension, or an empty string when the filename carries no tag
    (so untagged files all share one bucket and behave exactly as before).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    marker = stem.find("len-")
    return stem[marker:] if marker != -1 else ""


def record_to_slate(
    record: dict[str, Any],
    slate_size: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    """Convert one BGE-M3 mining record into a project slate record.

    BGE-M3 records look like ``{query, pos, neg, [pos_scores], [neg_scores]}``. We take
    one positive (highest ``pos_scores`` when present, else the first) and
    ``slate_size - 1`` negatives (top ``neg_scores`` when present, else the first ones),
    place the positive at index 0, and emit ``ranking = [1, 2, ..., slate_size]`` so the
    gold positive is rank 1. Returns ``None`` when the record cannot fill the slate.
    """
    positives = record.get("pos") or []
    negatives = record.get("neg") or []
    num_negatives = slate_size - 1
    if not positives or len(negatives) < num_negatives:
        return None

    pos_scores = record.get("pos_scores")
    if pos_scores and len(pos_scores) == len(positives):
        positive = positives[max(range(len(positives)), key=lambda i: pos_scores[i])]
    else:
        positive = rng.choice(positives)

    neg_scores = record.get("neg_scores")
    if neg_scores and len(neg_scores) == len(negatives):
        order = sorted(range(len(negatives)), key=lambda i: neg_scores[i], reverse=True)
        chosen = [negatives[i] for i in order[:num_negatives]]
    else:
        chosen = list(negatives)
        rng.shuffle(chosen)
        chosen = chosen[:num_negatives]

    documents = [positive, *chosen]
    ranking = list(range(1, slate_size + 1))
    return {"query": record["query"], "document": documents, "ranking": ranking}


def normalize_listwise_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize an E2Rank ``{query, document, ranking}`` record.

    ``ranking`` is a 1-indexed permutation of document positions, ordered from most
    to least relevant. Unlike BGE-M3 conversion, the complete candidate list and its
    teacher ordering are preserved so graded rewards and rank-based supervised losses
    consume exactly the same supervision.
    """
    query = record.get("query")
    documents = record.get("document")
    ranking = record.get("ranking")
    if not isinstance(query, str) or not query:
        raise ValueError("Listwise record requires a non-empty string field 'query'")
    if not isinstance(documents, list) or not documents or not all(
        isinstance(document, str) for document in documents
    ):
        raise ValueError("Listwise record requires a non-empty string list field 'document'")
    if (
        not isinstance(ranking, list)
        or len(ranking) != len(documents)
        or not all(isinstance(rank, int) and not isinstance(rank, bool) for rank in ranking)
    ):
        raise ValueError(
            "Listwise record requires an integer 'ranking' with the same length as 'document'"
        )
    expected = list(range(1, len(documents) + 1))
    if sorted(ranking) != expected:
        raise ValueError(
            f"Listwise ranking must be a 1-indexed permutation of {expected}, got {ranking}"
        )
    return {"query": query, "document": list(documents), "ranking": list(ranking)}


class EmbeddingDataset(Dataset):
    query_prompt_template = "Instruct: {task_description}\nQuery:{query}"

    def __init__(
        self,
        data_args: Any,
        batch_size: int | None = None,
        split: str = "train",
        query_prompt_template: str | None = None,
    ):
        if split not in {"train", "dev"}:
            raise ValueError(f"split must be 'train' or 'dev', got {split!r}")
        self.batch_size = batch_size or 32
        self.split = split
        if query_prompt_template is not None:
            self.query_prompt_template = query_prompt_template
        self.per_dataset_max_samples = data_args.per_dataset_max_samples
        self.dev_samples_per_source = getattr(data_args, "dev_samples_per_source", 0)
        self.slate_size = getattr(data_args, "slate_size", 8)
        # ``file_glob`` accepts a comma-separated list of patterns so several length
        # buckets can be mixed in one run, e.g. "*_len-0-500.jsonl,*_len-500-1000.jsonl".
        # A single-pattern string stays byte-identical to the legacy behaviour.
        raw_file_glob = getattr(data_args, "file_glob", "*_len-0-500.jsonl")
        self.file_glob = raw_file_glob
        self.file_globs = [g.strip() for g in raw_file_glob.split(",") if g.strip()]
        if not self.file_globs:
            raise ValueError(f"file_glob resolved to no patterns: {raw_file_glob!r}")
        # When multiple length buckets are read, batch each bucket separately so every
        # micro-batch holds documents of one length range (less padding waste). Off by
        # default: single-bucket runs are then bit-for-bit unchanged.
        self.batch_per_length_bucket = getattr(data_args, "batch_per_length_bucket", False)
        include_sources = getattr(data_args, "include_sources", None)
        self.include_sources = (
            {s.strip() for s in include_sources.split(",") if s.strip()}
            if include_sources
            else None
        )
        self.index_cache_dir = getattr(data_args, "index_cache_dir", None)

        # Lazy-loading state (L1): we keep only byte offsets in memory, not parsed rows.
        # ``_files`` holds per-file metadata (path, source). ``entries`` is the global
        # sample order after per-source batching + shuffling; each entry is an index into
        # ``_locations`` which stores (file_id, byte_offset). ``__getitem__`` seeks +
        # parses + converts on demand. Per-worker file handles are opened lazily in
        # ``_handle`` so DataLoader workers each get their own fd after fork.
        self._files: list[dict[str, Any]] = []
        self._locations: list[tuple[int, int]] = []
        self.entries: list[int] = []
        self._file_handles: dict[int, Any] = {}
        self._record_sources: dict[str, list[str]] = {}
        self._rng = random.Random()

        self._discover_files(data_args.data_path)
        self._build_index()

    # ------------------------------------------------------------------ discovery
    def _discover_files(self, data_path: str) -> None:
        """Populate ``self._files`` with (path, source, batch_key) for every data file.

        ``source`` drives the task prompt (length-agnostic); ``batch_key`` drives
        per-source batching and additionally splits by length bucket so every
        micro-batch holds documents of one length range, minimising padding waste.
        """
        if os.path.isfile(data_path):
            source = _source_name_from_dir(os.path.basename(os.path.dirname(data_path)))
            self._files.append(
                {
                    "path": data_path,
                    "source": source,
                    "batch_key": self._batch_key(source, data_path),
                    # A standalone E2Rank file may mix tasks and carries ``source`` per
                    # record. BGE-M3 direct-file inputs simply fall back to the parent.
                    "source_from_record": True,
                }
            )
            return

        if not os.path.isdir(data_path):
            raise FileNotFoundError(f"data_path does not exist: {data_path}")

        # Directory input: treat each immediate subdirectory as one source, reading the
        # files that match ``file_globs`` inside it. Files sitting directly under
        # data_path are also picked up (source inferred from the parent directory name).
        for entry in sorted(os.listdir(data_path)):
            full = os.path.join(data_path, entry)
            if os.path.isdir(full):
                if self.include_sources is not None and entry not in self.include_sources:
                    continue
                source = _source_name_from_dir(entry)
                # Union the matches across every glob, de-duplicating so overlapping
                # patterns never load the same file twice, then sort for stable order.
                matched: set[str] = set()
                for pattern in self.file_globs:
                    matched.update(glob.glob(os.path.join(full, pattern)))
                for path in sorted(matched):
                    self._files.append(
                        {
                            "path": path,
                            "source": source,
                            "batch_key": self._batch_key(source, path),
                            "source_from_record": False,
                        }
                    )
            elif entry.endswith(".jsonl") or entry.endswith(".json"):
                source = _source_name_from_dir(os.path.basename(data_path))
                self._files.append(
                    {
                        "path": full,
                        "source": source,
                        "batch_key": self._batch_key(source, full),
                        "source_from_record": True,
                    }
                )

        if not self._files:
            raise FileNotFoundError(
                f"No data files found under {data_path!r} (glob={self.file_glob!r})"
            )

    def _batch_key(self, source: str, path: str) -> str:
        """Batching key: ``source`` plus its length bucket when batching per length.

        With ``batch_per_length_bucket`` off (or a filename that carries no length tag)
        this collapses back to ``source``, so single-bucket runs are unchanged.
        """
        if not self.batch_per_length_bucket:
            return source
        bucket = _length_bucket_from_path(path)
        return f"{source}::{bucket}" if bucket else source

    # ------------------------------------------------------------------ indexing
    def _index_cache_path(self, path: str) -> str:
        if self.index_cache_dir:
            os.makedirs(self.index_cache_dir, exist_ok=True)
            safe = path.replace(os.sep, "__").lstrip("_")
            return os.path.join(self.index_cache_dir, safe + ".e2rank_idx.json")
        return path + ".e2rank_idx.json"

    def _scan_offsets(
        self,
        path: str,
        *,
        source_from_record: bool = False,
        fallback_source: str = "unknown",
    ) -> list[int]:
        """Return the byte offset of every line in ``path``, caching to disk.

        BGE-M3 source-directory files only need byte offsets. Standalone/mixed
        listwise files are parsed once while indexing so their per-record ``source``
        values can preserve single-source batches; those sources are cached beside the
        offsets. The cache is invalidated when file size or mtime changes.
        """
        cache_path = self._index_cache_path(path)
        stat = os.stat(path)
        signature = {"size": stat.st_size, "mtime": int(stat.st_mtime)}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                cached_sources = cached.get("sources")
                sources_are_usable = (
                    not source_from_record
                    or (
                        isinstance(cached_sources, list)
                        and len(cached_sources) == len(cached.get("offsets", []))
                    )
                )
                if cached.get("signature") == signature and sources_are_usable:
                    if source_from_record:
                        self._record_sources[path] = cached_sources
                    return cached["offsets"]
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        offsets: list[int] = []
        sources: list[str] = []
        with open(path, "rb") as f:
            offset = f.tell()
            line = f.readline()
            while line:
                if line.strip():
                    offsets.append(offset)
                    if source_from_record:
                        try:
                            record = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                            raise ValueError(
                                f"Invalid JSON record in {path} at byte offset {offset}"
                            ) from exc
                        raw_source = record.get("source") or fallback_source
                        sources.append(_source_name_from_dir(str(raw_source)))
                offset = f.tell()
                line = f.readline()
        if source_from_record:
            self._record_sources[path] = sources
        try:
            with open(cache_path, "w") as f:
                cached_index = {"signature": signature, "offsets": offsets}
                if source_from_record:
                    cached_index["sources"] = sources
                json.dump(cached_index, f)
        except OSError as exc:
            print(f"Warning: could not write offset cache {cache_path}: {exc}")
        return offsets

    def _build_index(self) -> None:
        """Build the per-source-capped, per-batch-key batched, shuffled sample order.

        Two grouping levels are in play. The training cap and dev split apply per
        **source** so ``per_dataset_max_samples`` keeps its "total per task" meaning
        regardless of how many length buckets a source spans. Batching then happens per
        **batch_key** (source + length bucket when ``batch_per_length_bucket``) so each
        micro-batch stays single-source *and* single-length. When batching is off, or a
        source has only one bucket, ``batch_key == source`` and this is identical to the
        previous behaviour.
        """
        locations_by_source: dict[str, list[int]] = defaultdict(list)
        batch_key_by_location: dict[int, str] = {}
        for file_id, meta in enumerate(self._files):
            offsets = self._scan_offsets(
                meta["path"],
                source_from_record=meta["source_from_record"],
                fallback_source=meta["source"],
            )
            record_sources = self._record_sources.get(meta["path"])
            for record_index, offset in enumerate(offsets):
                source = (
                    record_sources[record_index]
                    if record_sources is not None
                    else meta["source"]
                )
                batch_key = (
                    self._batch_key(source, meta["path"])
                    if meta["source_from_record"]
                    else meta["batch_key"]
                )
                location_id = len(self._locations)
                self._locations.append((file_id, offset))
                locations_by_source[source].append(location_id)
                batch_key_by_location[location_id] = batch_key

        ordered_batches: list[list[int]] = []
        for source, location_ids in locations_by_source.items():
            # Seeded by the caller's set_seed(), so the same seed yields the same split.
            random.shuffle(location_ids)
            # Carve the dev slice off the FRONT, before the training cap, so changing
            # per_dataset_max_samples can never leak a dev example into training.
            if self.dev_samples_per_source > 0:
                dev_ids = location_ids[: self.dev_samples_per_source]
                train_ids = location_ids[self.dev_samples_per_source :]
            else:
                dev_ids, train_ids = [], location_ids
            if self.split == "dev":
                limited_ids = dev_ids
            else:
                limited_ids = (
                    train_ids
                    if self.per_dataset_max_samples is None
                    else train_ids[: self.per_dataset_max_samples]
                )
            # The per-source cap is applied above; now split the survivors by batch_key
            # so each length bucket forms its own uniform-length batches. The trailing
            # partial batch of each bucket is dropped (as before, per group).
            ids_by_batch_key: dict[str, list[int]] = defaultdict(list)
            for location_id in limited_ids:
                ids_by_batch_key[batch_key_by_location[location_id]].append(location_id)
            for batch_key, bucket_ids in ids_by_batch_key.items():
                for start in range(0, len(bucket_ids), self.batch_size):
                    batch = bucket_ids[start : start + self.batch_size]
                    if len(batch) == self.batch_size:
                        ordered_batches.append(batch)
                    else:
                        print(f"Skip 1 {self.split} batch for dataset {batch_key}.")

        random.shuffle(ordered_batches)
        self.entries = [loc_id for batch in ordered_batches for loc_id in batch]
        self.num_batches = len(ordered_batches)
        print(
            f"Indexed {len(self.entries)} {self.split} samples in "
            f"{self.num_batches} single-source batches "
            f"across {len(self._files)} file(s)."
        )

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.entries)

    def _format_query(self, task_name: str, query: str) -> str:
        retrieval_prompt = TASK_PROMPTS.get(task_name, DEFAULT_TASK_PROMPTS)
        return format_embedding_text(
            self.query_prompt_template,
            query,
            task_description=retrieval_prompt,
        )

    def _handle(self, file_id: int):
        handle = self._file_handles.get(file_id)
        if handle is None:
            handle = open(self._files[file_id]["path"], "r")
            self._file_handles[file_id] = handle
        return handle

    def close(self) -> None:
        for handle in self._file_handles.values():
            handle.close()
        self._file_handles.clear()

    def __del__(self):
        # Dataset workers own independent lazy handles after fork. Closing whichever
        # handles belong to this instance avoids leaking descriptors in short probes and
        # tests while leaving normal DataLoader lifetime unchanged.
        try:
            self.close()
        except Exception:
            pass

    def _read_record(self, location_id: int) -> dict[str, Any]:
        file_id, offset = self._locations[location_id]
        handle = self._handle(file_id)
        handle.seek(offset)
        return json.loads(handle.readline())

    def _convert(self, file_id: int, record: dict[str, Any]) -> dict[str, Any] | None:
        if "document" in record or "ranking" in record:
            if "document" not in record or "ranking" not in record:
                raise ValueError(
                    "Listwise records must contain both 'document' and 'ranking'"
                )
            converted = normalize_listwise_record(record)
        elif "pos" in record or "neg" in record:
            converted = record_to_slate(record, self.slate_size, self._rng)
        else:
            raise ValueError(
                "Unsupported training record schema: expected either "
                "{query, document, ranking} or {query, pos, neg}"
            )
        if converted is None:
            return None
        raw_source = record.get("source") or self._files[file_id]["source"]
        source = _source_name_from_dir(str(raw_source))
        converted["query"] = self._format_query(source, converted["query"])
        return converted

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Resolve within the same single-source batch so a dropped BGE-M3 record (too
        # few negatives to fill the slate) is replaced by another sample sharing the
        # batch's source, keeping the batch single-source and the slate length uniform.
        batch_start = (index // self.batch_size) * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.entries))
        order = [index] + [i for i in range(batch_start, batch_end) if i != index]
        for candidate in order:
            location_id = self.entries[candidate]
            file_id, _ = self._locations[location_id]
            record = self._read_record(location_id)
            converted = self._convert(file_id, record)
            if converted is not None:
                return converted
        raise RuntimeError(
            f"No convertible sample in batch starting at {batch_start}; "
            f"slate_size={self.slate_size} may exceed available negatives."
        )



class SingleSourceBatchSampler(Sampler[int]):
    """Sample-level sampler that preserves ``EmbeddingDataset``'s per-source batching.

    ``EmbeddingDataset`` lays its samples out as consecutive blocks of ``batch_size``
    drawn from a single source, so that every micro-batch shares one task prompt and
    in-batch negatives stay in-domain. The HF Trainer's default ``RandomSampler``
    shuffles at the *sample* level and silently destroys that layout, mixing every
    source into every batch. This sampler shuffles whole blocks instead, keeping the
    intra-block order intact.

    The block permutation is derived from ``seed + epoch`` only, so every rank walks
    the same global batch order; accelerate then hands whole batches to ranks
    round-robin (``split_batches=False``), and each rank still sees single-source
    micro-batches.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        num_samples = len(self.dataset)
        num_blocks = num_samples // self.batch_size
        if self.shuffle and num_blocks > 1:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            block_order = torch.randperm(num_blocks, generator=generator).tolist()
        else:
            block_order = range(num_blocks)

        for block in block_order:
            yield from range(block * self.batch_size, (block + 1) * self.batch_size)
        # EmbeddingDataset drops partial per-source batches, so this is normally empty.
        yield from range(num_blocks * self.batch_size, num_samples)


def build_relevance_labels(
    ranking: torch.Tensor,
    scheme: str = "graded",
) -> torch.Tensor:
    if ranking is None:
        raise ValueError("ranking is required to build relevance labels")
    if ranking.dim() != 2:
        raise ValueError(f"ranking must be a 2D tensor, got shape {tuple(ranking.shape)}")
    if scheme not in {"graded", "binary"}:
        raise ValueError(f"Unsupported relevance scheme: {scheme}")

    batch_size, slate_length = ranking.shape
    relevance = torch.zeros(batch_size, slate_length, device=ranking.device, dtype=torch.float32)

    rank_scores = torch.zeros(slate_length, device=ranking.device, dtype=torch.float32)
    if slate_length > 0:
        rank_scores[0] = 3.0 if scheme == "graded" else 1.0
    if scheme == "graded":
        if slate_length > 1:
            rank_scores[1:min(5, slate_length)] = 2.0
        if slate_length > 5:
            rank_scores[5:min(10, slate_length)] = 1.0

    relevance.scatter_(
        dim=1,
        index=ranking,
        src=rank_scores.unsqueeze(0).expand(batch_size, -1),
    )
    return relevance


def build_rank_labels(ranking: torch.Tensor) -> torch.Tensor:
    """Invert a 0-indexed teacher permutation into dense higher-is-better labels."""
    if ranking.dim() != 2:
        raise ValueError(f"ranking must be a 2D tensor, got shape {tuple(ranking.shape)}")
    batch_size, slate_length = ranking.shape
    labels = torch.zeros_like(ranking, dtype=torch.float32)
    rank_values = torch.arange(
        slate_length,
        0,
        -1,
        device=ranking.device,
        dtype=torch.float32,
    )
    labels.scatter_(
        dim=1,
        index=ranking,
        src=rank_values.unsqueeze(0).expand(batch_size, -1),
    )
    return labels


def build_slate_inputs(
    positive_document: Dict[str, torch.Tensor],
    negative_document: Dict[str, torch.Tensor],
    batch_size: int,
    slate_length: int,
) -> Dict[str, torch.Tensor]:
    """Re-interleave the collator's split tensors into one flat ``[batch * slate, ...]`` batch.

    ``EmbeddingDataCollator`` emits every sample's gold positive in ``positive_document``
    (``[batch, ...]``) and all negatives in ``negative_document``, ordered sample-major
    (``[batch * (slate - 1), ...]``). Encoding them needs a single flat batch whose rows read
    ``(sample 0 positive, sample 0 negatives..., sample 1 positive, ...)`` so that the
    ``reshape(batch, slate, -1)`` on the far side puts each sample's own candidates into its
    own slate -- lining up with the ``relevance_labels`` the collator built in that same order.

    A plain ``cat((positive, negative), dim=0)`` does *not* satisfy this: it lays out all
    positives first, so ``reshape`` would deal other samples' positives into sample 0's slate.
    """
    num_negatives = slate_length - 1
    slate_inputs: Dict[str, torch.Tensor] = {}
    for key, positive_value in positive_document.items():
        negative_value = negative_document[key].reshape(batch_size, num_negatives, -1)
        slate_value = torch.cat((positive_value.unsqueeze(1), negative_value), dim=1)
        slate_inputs[key] = slate_value.reshape(batch_size * slate_length, -1)
    return slate_inputs


class EmbeddingDataCollator:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        query_max_length: int = 512,
        doc_max_length: int = 1024,
        relevance_scheme: str = "binary",
        document_prompt_template: str = "{document}",
        append_token: str = "pad",
        **_: Any,
    ):
        if relevance_scheme not in {"binary", "graded"}:
            raise ValueError(f"Unsupported relevance_scheme: {relevance_scheme}")
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.relevance_scheme = relevance_scheme
        self.document_prompt_template = document_prompt_template
        self.append_token = append_token
        if not self.tokenizer.pad_token:
            if getattr(self.tokenizer, "eot_token", None):
                self.tokenizer.pad_token = self.tokenizer.eot_token
            elif self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.bos_token
        if not self.tokenizer.pad_token:
            raise ValueError(
                "Tokenizer has no pad/eot/eos/bos token available for batched embedding inputs"
            )
        print(f"use ``{self.tokenizer.pad_token}`` as pad token for llm")

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        queries = append_configured_token(
            [instance["query"] for instance in instances],
            self.tokenizer,
            self.append_token,
        )
        query_inputs = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )

        ranking = torch.tensor([instance["ranking"] for instance in instances], dtype=torch.long) - 1
        original_relevance_labels = build_relevance_labels(
            ranking=ranking,
            scheme=self.relevance_scheme,
        )
        # Dense teacher-order targets for rank-based supervised objectives. ``ranking``
        # maps rank position -> original document index; scatter inverts that mapping
        # and assigns a larger value to every better-ranked document. Unlike the
        # possibly coarsened relevance labels, these preserve the full permutation.
        original_rank_labels = build_rank_labels(ranking)

        positive_documents: list[str] = []
        negative_documents: list[str] = []
        relevance_labels: list[torch.Tensor] = []
        rank_labels: list[torch.Tensor] = []
        for sample_idx, instance in enumerate(instances):
            documents_for_sample = instance["document"]
            positive_index = int(ranking[sample_idx, 0].item())
            negative_indices = [
                document_idx
                for document_idx in range(len(documents_for_sample))
                if document_idx != positive_index
            ]

            positive_documents.append(documents_for_sample[positive_index])
            negative_documents.extend(documents_for_sample[document_idx] for document_idx in negative_indices)
            ordered_indices = torch.tensor(
                [positive_index, *negative_indices],
                device=original_relevance_labels.device,
                dtype=torch.long,
            )
            relevance_labels.append(original_relevance_labels[sample_idx].gather(dim=0, index=ordered_indices))
            rank_labels.append(original_rank_labels[sample_idx].gather(dim=0, index=ordered_indices))

        documents = [
            format_embedding_text(self.document_prompt_template, document)
            for document in [*positive_documents, *negative_documents]
        ]
        documents = append_configured_token(documents, self.tokenizer, self.append_token)
        document_inputs = self.tokenizer(
            documents,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )

        batch_size = len(instances)
        return {
            "query": query_inputs,
            "positive_document": {
                key: value[:batch_size]
                for key, value in document_inputs.items()
            },
            "negative_document": {
                key: value[batch_size:]
                for key, value in document_inputs.items()
            },
            "relevance_labels": torch.stack(relevance_labels),
            "rank_labels": torch.stack(rank_labels),
        }
