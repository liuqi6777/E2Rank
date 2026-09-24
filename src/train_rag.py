"""Train query-only RAG controls against an immutable external corpus index."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import HfArgumentParser, Trainer as HFTrainer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from config import LoraArguments, ModelArguments, TrainingArguments
from grpo_trainer import EmbeddingTrainerMixin
from rag.config import (
    GENERATION_REWARDS,
    RAGDatasetArguments,
    RAGGeneratorArguments,
    RAGIndexArguments,
    RAGRewardArguments,
)
from rag.data import CandidateManifestDataset, RAGQueryCollator
from rag.generator import FrozenGeneratorClient
from rag.index import FrozenDistributedIndex, sha256_file
from rag.models import RAGRLModel, RAGSupervisedModel
from rag.shortlist_rl import RAGShortlistRLModel
from rag.protocol import validate_query_index_protocol
from rag.retrieval_probe import RAGRetrievalProbeCallback, load_probe_queries
from rag.tuning_eval import RAGTuningEvalCallback
from train import (
    apply_gradient_checkpointing,
    guard_output_dir,
    load_backbone_and_tokenizer,
    parse_arguments,
    save_run_artifacts,
    setup_logging,
    shutdown_distributed,
)


logger = logging.getLogger(__name__)
RAG_CONFIG_SLOTS = ("train", "model", "rag")


class RAGTrainer(EmbeddingTrainerMixin, HFTrainer):
    def _extra_train_metrics(self, outputs):
        return self._output_field(outputs, "exploration_metrics") or {}

    train_metric_names = ("reward_mean", "reward_std", "degenerate_fraction", "span_rank")
    train_metric_log_names = {
        "reward_mean": "reward/mean",
        "reward_std": "reward/std",
        "degenerate_fraction": "advantages/degenerate_frac",
        # Watch this under conditional projection: if the span saturates at the
        # embedding dimension the projector is the identity and CP has silently
        # degenerated into the score-function estimator.
        "span_rank": "projection/query_span_rank_mean",
    }


def _validate_protocol(
    lora_args: LoraArguments,
    data_args: RAGDatasetArguments,
    reward_args: RAGRewardArguments,
    generator_args: RAGGeneratorArguments,
) -> None:
    if lora_args.lora_enabled:
        raise ValueError("RAG training uses full query-encoder fine-tuning; set lora_enabled=false")
    if lora_args.lora_path:
        raise ValueError(
            "RAG methods must initialize from E0; use Trainer resume checkpoints only to resume the same run"
        )
    if not data_args.rag_candidate_manifest:
        raise ValueError("rag_candidate_manifest is required")
    if generator_args.rag_generator_top_k != reward_args.rag_retrieval_k:
        raise ValueError(
            "G3 training requires rag_generator_top_k == rag_retrieval_k so all "
            "rewarded passages are visible to the generator"
        )


def _assert_full_backbone_trainable(backbone) -> None:
    trainable = [name for name, parameter in backbone.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Query encoder has no trainable parameters")
    frozen = [name for name, parameter in backbone.named_parameters() if not parameter.requires_grad]
    if frozen:
        raise RuntimeError(
            "RAG full fine-tuning requires the complete query encoder to be trainable; "
            "frozen parameters: " + ", ".join(frozen[:10])
        )


def _validate_candidate_manifest(data_args: RAGDatasetArguments, index: FrozenDistributedIndex) -> None:
    candidate_path = Path(data_args.rag_candidate_manifest).resolve()
    metadata_path = candidate_path.with_suffix(candidate_path.suffix + ".manifest.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing candidate metadata manifest: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("artifact_origin") != "E2Rank-RL":
        raise ValueError("Candidate manifest was not produced by this project")
    if metadata.get("candidate_manifest_sha256") != sha256_file(candidate_path):
        raise ValueError("Candidate manifest content hash does not match its metadata")
    if metadata.get("index_manifest_sha256") != sha256_file(index.manifest_path):
        raise ValueError("Candidate pool was mined against a different frozen index")
    if metadata.get("source_manifest_sha256") != index.manifest.get("source_manifest_sha256"):
        raise ValueError("Candidate pool and frozen index use different FlashRAG source manifests")
    label_protocol = metadata.get("training_label_protocol", {})
    if label_protocol.get("type") != "external_binary_passage_qrels":
        raise ValueError("Candidate pool does not use the fixed binary passage qrels protocol")
    qrels_path = Path(metadata.get("qrels", ""))
    if not qrels_path.is_absolute():
        qrels_path = metadata_path.parent / qrels_path
    qrels_manifest_path = Path(metadata.get("qrels_manifest", ""))
    if not qrels_manifest_path.is_absolute():
        qrels_manifest_path = metadata_path.parent / qrels_manifest_path
    if not qrels_path.is_file() or metadata.get("qrels_sha256") != sha256_file(qrels_path):
        raise ValueError("Candidate pool references missing or changed qrels")
    if (
        not qrels_manifest_path.is_file()
        or metadata.get("qrels_manifest_sha256") != sha256_file(qrels_manifest_path)
    ):
        raise ValueError("Candidate pool references a missing or changed qrels manifest")
    with qrels_manifest_path.open("r", encoding="utf-8") as handle:
        qrels_manifest = json.load(handle)
    if qrels_manifest.get("qrels_sha256") != metadata.get("qrels_sha256"):
        raise ValueError("Candidate pool and qrels manifest disagree on qrels contents")
    if (
        qrels_manifest.get("inputs", {}).get("corpus", {}).get("sha256")
        != index.manifest.get("corpus_sha256")
    ):
        raise ValueError("Candidate qrels and frozen index use different corpus contents")
    if int(metadata.get("depth", -1)) != data_args.rag_candidate_depth:
        raise ValueError("Candidate manifest depth does not match rag_candidate_depth")
    if set(metadata.get("statistics", {})) != set(data_args.train_sources):
        raise ValueError("Candidate sources do not match rag_train_sources")


def _write_rag_run_manifest(
    output_dir: str,
    data_args: RAGDatasetArguments,
    index: FrozenDistributedIndex,
    initial_index_hash: str,
    elapsed_seconds: float,
    peak_gpu_memory_gib: float,
    generator_statistics: dict | None,
    generator_gpu_count: int,
) -> None:
    index.verify()
    final_index_hash = sha256_file(index.manifest_path)
    if final_index_hash != initial_index_hash:
        raise RuntimeError("Frozen index manifest changed during training")
    candidate_path = Path(data_args.rag_candidate_manifest).resolve()
    payload = {
        "format_version": 1,
        "artifact_origin": "E2Rank-RL",
        "index_manifest": index.manifest_path,
        "index_manifest_sha256_before": initial_index_hash,
        "index_manifest_sha256_after": final_index_hash,
        "corpus_sha256_before": index.manifest.get("corpus_sha256"),
        "corpus_sha256_after": index.manifest.get("corpus_sha256"),
        "corpus_reencoding_count": 0,
        "index_rebuild_count": 0,
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "generator_manifest_hash": generator_statistics.get("generator_manifest_hash") if generator_statistics else None,
        "telemetry": {
            "wall_hours": elapsed_seconds / 3600,
            "query_index_gpu_hours": elapsed_seconds * index.world_size / 3600 if torch.cuda.is_available() else 0.0,
            "generator_gpu_hours": elapsed_seconds * generator_gpu_count / 3600 if generator_statistics else 0.0,
            "gpu_hours": elapsed_seconds * (index.world_size + (generator_gpu_count if generator_statistics else 0)) / 3600 if torch.cuda.is_available() else 0.0,
            "peak_gpu_memory_gib": peak_gpu_memory_gib,
            **(generator_statistics or {"generation_requests": 0}),
        },
    }
    with open(Path(output_dir) / "rag_run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> None:
    parser = HfArgumentParser((
        ModelArguments,
        TrainingArguments,
        LoraArguments,
        RAGDatasetArguments,
        RAGIndexArguments,
        RAGRewardArguments,
        RAGGeneratorArguments,
    ))
    (
        model_args,
        training_args,
        lora_args,
        data_args,
        index_args,
        reward_args,
        generator_args,
    ) = parse_arguments(parser, base_slots=RAG_CONFIG_SLOTS)
    _validate_protocol(lora_args, data_args, reward_args, generator_args)
    training_args.remove_unused_columns = False
    guard_output_dir(training_args)
    setup_logging(training_args, {
        "RAG data": data_args,
        "RAG index": index_args,
        "RAG reward": reward_args,
        "RAG generator": generator_args,
    })
    set_seed(training_args.seed)

    # Accessing TrainingArguments.device initializes the torchrun process group
    # before index shards are assigned to ranks.
    training_device = training_args.device
    device = (
        training_device
        if index_args.rag_index_device == "cuda"
        else torch.device(index_args.rag_index_device)
    )
    if device != training_device:
        raise ValueError(
            f"The RAG index and query encoder must share a device during training: "
            f"index={device}, query={training_device}"
        )
    index = FrozenDistributedIndex(
        index_args.rag_index_manifest,
        backend=index_args.rag_index_backend,
        device=device,
        verify_hashes=index_args.rag_index_verify_hashes,
        search_batch_size=index_args.rag_index_search_batch_size,
    )
    _validate_candidate_manifest(data_args, index)
    initial_index_hash = sha256_file(index.manifest_path)
    backbone, tokenizer = load_backbone_and_tokenizer(model_args, lora_args)
    _assert_full_backbone_trainable(backbone)
    base_config = getattr(backbone, "config", None)
    validate_query_index_protocol(
        index.manifest,
        model_name_or_path=model_args.model_name_or_path,
        resolved_model_revision=getattr(base_config, "_commit_hash", None),
        pooling_method=model_args.pooling_method,
        padding_side=model_args.padding_side,
        append_token=model_args.append_token,
    )

    generator = None
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if reward_args.rag_retrieval_reward in GENERATION_REWARDS and rank == 0:
        generator = FrozenGeneratorClient(
            endpoint=generator_args.rag_generator_endpoint or "",
            model=generator_args.rag_generator_model,
            revision=generator_args.rag_generator_revision,
            cache_path="",
            max_input_length=generator_args.rag_generator_max_input_length,
            max_new_tokens=generator_args.rag_generator_max_new_tokens,
            timeout_seconds=generator_args.rag_generator_timeout_seconds,
            system_prompt=generator_args.rag_generator_system_prompt,
        )

    if data_args.rag_objective in {"infonce", "ranknet", "lambdaloss"}:
        model = RAGSupervisedModel(
            backbone,
            index,
            objective=data_args.rag_objective,
            temperature=data_args.rag_temperature,
            pooling_method=model_args.pooling_method,
            relevance_scheme=data_args.rag_relevance_scheme,
            anchor_coef=data_args.rag_anchor_coef,
            ndcg_k=reward_args.rag_retrieval_k,
            use_in_batch_candidates=data_args.rag_use_in_batch_candidates,
            in_batch_include_negatives=data_args.rag_in_batch_include_negatives,
            cross_device_negatives=data_args.rag_cross_device_negatives,
            in_batch_pool_size=data_args.rag_in_batch_pool_size,
        )
    elif data_args.rag_objective == "shortlist_rl":
        model = RAGShortlistRLModel(
            backbone,
            index,
            relevance_scheme=data_args.rag_relevance_scheme,
            reward_k=reward_args.rag_retrieval_k,
            slate_size=reward_args.rag_slate_size,
            shortlist_size=reward_args.rag_shortlist_size,
            group_size=reward_args.rag_group_size,
            kappa=reward_args.rag_kappa,
            pooling_method=model_args.pooling_method,
            gradient_estimator=reward_args.rag_gradient_estimator,
            advantage_baseline=reward_args.rag_advantage_baseline,
            advantage_norm=reward_args.rag_advantage_norm,
            target_alignment=reward_args.rag_target_alignment,
            final_alignment=reward_args.rag_final_alignment,
            exploration_schedule=reward_args.rag_exploration_schedule,
            cross_device_negatives=data_args.rag_cross_device_negatives,
            anchor_coef=data_args.rag_anchor_coef,
        )
    else:
        model = RAGRLModel(
            backbone,
            index,
            reward_type=reward_args.rag_retrieval_reward,
            retrieval_k=reward_args.rag_retrieval_k,
            group_size=reward_args.rag_group_size,
            kappa=reward_args.rag_kappa,
            pooling_method=model_args.pooling_method,
            generator=generator,
            generator_top_k=generator_args.rag_generator_top_k,
            normalize_advantages=reward_args.rag_advantage_normalize,
            advantage_baseline=reward_args.rag_advantage_baseline,
            advantage_norm=reward_args.rag_advantage_norm,
            advantage_baseline_momentum=reward_args.rag_advantage_baseline_momentum,
            target_alignment=reward_args.rag_target_alignment,
            final_alignment=reward_args.rag_final_alignment,
            exploration_schedule=reward_args.rag_exploration_schedule,
            relevance_scheme=data_args.rag_relevance_scheme,
            anchor_coef=data_args.rag_anchor_coef,
        )
    model.train()
    apply_gradient_checkpointing(model, training_args, lora_args)

    train_dataset = CandidateManifestDataset(
        data_args.rag_candidate_manifest,
        split=data_args.rag_split,
        tuning_fraction=data_args.rag_tuning_fraction,
        tuning_seed=data_args.rag_tuning_seed,
        query_prompt_template=model_args.query_prompt_template,
        max_samples=data_args.rag_max_train_samples,
    )
    collator = RAGQueryCollator(
        tokenizer,
        query_max_length=min(data_args.rag_query_max_length, model_args.embedding_max_length),
        append_token=model_args.append_token,
    )

    # E0 anchors for the drift penalty (rag_anchor_coef > 0). The precompute
    # runs at on_train_begin, not here: the backbone was loaded under DeepSpeed's
    # ZeRO-3 init context and its parameters are partitioned until the Trainer
    # wraps it (a plain forward raises "'weight' must be 2-D"). The callback
    # encodes through the wrapped model, whose forward gathers on the fly, and
    # no optimizer step has run at that point so the weights are still the
    # initialization the anchors must capture.
    anchor_callback = None
    if data_args.rag_anchor_coef > 0:
        from rag.anchors import RAGAnchorPrecomputeCallback, anchor_cache_path

        cache_dir = Path(data_args.rag_candidate_manifest).parent / "anchors"
        cache_dir.mkdir(parents=True, exist_ok=True)
        # One cache per (manifest, split, tuning split, sample cap, run): the
        # callback unlinks and rewrites its file, so two anchor arms launched
        # together must not resolve to the same path -- one trial's rank 0
        # would truncate the file mid-write out from under the other's ranks,
        # and the loser would attach to a half-written file of zero rows.
        variant = (
            f"{data_args.rag_split}:{data_args.rag_tuning_fraction}:"
            f"{data_args.rag_tuning_seed}:{data_args.rag_max_train_samples}:"
            f"{training_args.output_dir}"
        )
        anchor_callback = RAGAnchorPrecomputeCallback(
            dataset=train_dataset,
            tokenizer=tokenizer,
            training_args=training_args,
            data_args=data_args,
            model_args=model_args,
            dimension=backbone.config.hidden_size,
            vector_path=anchor_cache_path(data_args.rag_candidate_manifest, cache_dir, variant),
        )
    resume_checkpoint = None
    if not training_args.overwrite_output_dir and os.path.isdir(training_args.output_dir):
        resume_checkpoint = get_last_checkpoint(training_args.output_dir)

    # Monitoring only. The tuning split is held out whenever rag_split is
    # "train"; under rag_split "full" it is in-sample and the numbers read as a
    # drift probe rather than generalization.
    tuning_callback = None
    if data_args.rag_tuning_eval:
        tuning_dataset = CandidateManifestDataset(
            data_args.rag_candidate_manifest,
            split="tuning",
            tuning_fraction=data_args.rag_tuning_fraction,
            tuning_seed=data_args.rag_tuning_seed,
            query_prompt_template=model_args.query_prompt_template,
            max_samples=data_args.rag_tuning_eval_samples,
        )
        tuning_callback = RAGTuningEvalCallback(
            model=model,
            index=index,
            dataset=tuning_dataset,
            collator=collator,
            batch_size=training_args.per_device_train_batch_size,
        )

    # Measures retrieval rather than re-ranking, and separates in-domain from
    # held-out datasets. Round 2's re-ranking probe rose while the real metric
    # fell, so this is the one that can be trusted for model selection.
    retrieval_callback = None
    if data_args.rag_retrieval_probe:
        retrieval_callback = RAGRetrievalProbeCallback(
            model=model,
            index=index,
            tokenizer=tokenizer,
            queries=load_probe_queries(
                Path(data_args.rag_dataset_root).parent / "flashrag_manifest.json",
                per_dataset=data_args.rag_retrieval_probe_per_dataset,
            ),
            retrieval_k=data_args.rag_retrieval_probe_k,
            batch_size=training_args.per_device_train_batch_size,
            query_max_length=data_args.rag_query_max_length,
            append_token=model_args.append_token,
            query_prompt_template=model_args.query_prompt_template,
        )

    from grpo_trainer import restore_exploration_state
    restore_exploration_state(model, resume_checkpoint)
    callbacks = [
        cb
        for cb in (tuning_callback, retrieval_callback, anchor_callback)
        if cb is not None
    ]
    trainer = RAGTrainer(
        model_args=model_args,
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks or None,
    )
    for callback in callbacks:
        callback.trainer = trainer
    training_started = time.perf_counter()
    trainer.train(resume_from_checkpoint=True if resume_checkpoint else None)
    elapsed_seconds = time.perf_counter() - training_started
    peak_gpu_memory_gib = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    if dist.is_available() and dist.is_initialized():
        telemetry = torch.tensor(
            [elapsed_seconds, peak_gpu_memory_gib], device=training_args.device, dtype=torch.float64
        )
        dist.all_reduce(telemetry, op=dist.ReduceOp.MAX)
        elapsed_seconds, peak_gpu_memory_gib = telemetry.tolist()
    save_run_artifacts(
        trainer,
        training_args,
        tokenizer,
        model_args=model_args,
        rag_data_args=data_args,
        rag_index_args=index_args,
        rag_reward_args=reward_args,
        rag_generator_args=generator_args,
    )
    if trainer.is_world_process_zero():
        _write_rag_run_manifest(
            training_args.output_dir,
            data_args,
            index,
            initial_index_hash,
            elapsed_seconds,
            peak_gpu_memory_gib,
            generator.statistics() if generator is not None else None,
            generator_args.rag_generator_gpu_count,
        )
    if generator is not None:
        generator.close()
    index.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_distributed()
