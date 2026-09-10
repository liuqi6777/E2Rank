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
    RAGDatasetArguments,
    RAGGeneratorArguments,
    RAGIndexArguments,
    RAGRewardArguments,
)
from rag.data import CandidateManifestDataset, RAGQueryCollator
from rag.generator import FrozenGeneratorClient
from rag.index import FrozenDistributedIndex, sha256_file
from rag.models import RAGRLModel, RAGSupervisedModel
from rag.protocol import validate_query_index_protocol
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
    train_metric_names = ("reward_mean", "reward_std", "degenerate_fraction")
    train_metric_log_names = {
        "reward_mean": "reward/mean",
        "reward_std": "reward/std",
        "degenerate_fraction": "advantages/degenerate_frac",
    }


def _validate_protocol(lora_args: LoraArguments, data_args: RAGDatasetArguments) -> None:
    if lora_args.lora_enabled:
        raise ValueError("RAG training uses full query-encoder fine-tuning; set lora_enabled=false")
    if lora_args.lora_path:
        raise ValueError(
            "RAG methods must initialize from E0; use Trainer resume checkpoints only to resume the same run"
        )
    if not data_args.rag_candidate_manifest:
        raise ValueError("rag_candidate_manifest is required")


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
    _validate_protocol(lora_args, data_args)
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
    if reward_args.rag_retrieval_reward == "answer_f1" and rank == 0:
        generator = FrozenGeneratorClient(
            endpoint=generator_args.rag_generator_endpoint or "",
            model=generator_args.rag_generator_model,
            revision=generator_args.rag_generator_revision,
            cache_path=generator_args.rag_generator_cache,
            max_input_length=generator_args.rag_generator_max_input_length,
            max_new_tokens=generator_args.rag_generator_max_new_tokens,
            timeout_seconds=generator_args.rag_generator_timeout_seconds,
            system_prompt=generator_args.rag_generator_system_prompt,
        )

    if data_args.rag_objective in {"infonce", "ranknet"}:
        model = RAGSupervisedModel(
            backbone,
            index,
            objective=data_args.rag_objective,
            temperature=data_args.rag_temperature,
            pooling_method=model_args.pooling_method,
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
    resume_checkpoint = None
    if not training_args.overwrite_output_dir and os.path.isdir(training_args.output_dir):
        resume_checkpoint = get_last_checkpoint(training_args.output_dir)

    trainer = RAGTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
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
