"""Prepare pinned ReasonRank data retaining all positives and variable candidates.

uv run --no-project --with pyarrow python scripts/prepare_reasonrank.py
Raw parquet and previous conversion outputs are never overwritten.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import unicodedata
from typing import Any, Iterator

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PASSAGE_MARKER = re.compile(r"(?m)^\[(\d+)\]\s+")
LABEL_FORMAT = re.compile(r"^\s*\[\d+\](?:\s*>\s*\[\d+\])*\s*$")
SEARCH_QUERY_MARKER = "\nSearch Query: "
RANK_INSTRUCTION_MARKER = "\nRank the "


def iter_parquet_rows(path: Path, batch_size: int = 128) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _user_prompt(messages: Any) -> str:
    if not isinstance(messages, list):
        raise ValueError("'prompt' must be a list of chat messages")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content:
                return content
            break
    raise ValueError("'prompt' does not contain a non-empty user message")


def parse_query_and_passages(prompt: str) -> tuple[str, list[str]]:
    passage_section, separator, footer = prompt.rpartition(SEARCH_QUERY_MARKER)
    if not separator:
        raise ValueError("user prompt is missing the final 'Search Query:' marker")

    query, separator, _ = footer.partition(RANK_INSTRUCTION_MARKER)
    query = query.strip()
    if not separator or not query:
        raise ValueError("user prompt has a malformed final query/ranking instruction")

    matches = list(PASSAGE_MARKER.finditer(passage_section))
    passage_numbers = [int(match.group(1)) for match in matches]
    expected_numbers = list(range(1, len(matches) + 1))
    if passage_numbers != expected_numbers:
        raise ValueError(
            f"passage identifiers must be consecutive from 1, got {passage_numbers}"
        )

    passages = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(passage_section)
        passage = passage_section[match.end() : end].strip()
        if not passage:
            raise ValueError(f"passage [{index + 1}] is empty")
        passages.append(passage)
    return query, passages


def parse_teacher_ranking(label: Any, candidate_count: int) -> list[int]:
    if not isinstance(label, str) or not LABEL_FORMAT.fullmatch(label):
        raise ValueError(f"malformed teacher label: {label!r}")
    ranking = [int(value) for value in re.findall(r"\[(\d+)\]", label)]
    expected = list(range(1, candidate_count + 1))
    if sorted(ranking) != expected:
        raise ValueError(
            f"teacher label must be a permutation of {expected}, got {ranking}"
        )
    return ranking


def validate_record(record):
    docs, ids, relevance = record['document'], record['document_ids'], record['relevance']
    if not isinstance(record['query'], str) or not record['query'].strip():
        raise ValueError('Empty query')
    if len(docs) < 2 or len(ids) != len(docs) or len(relevance) != len(docs):
        raise ValueError('Candidate dimensions must agree and include a negative')
    if not all(isinstance(d, str) and d.strip() for d in docs):
        raise ValueError('Stored candidates must be nonempty; padding belongs to collation')
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError('Document IDs must be unique nonempty strings')
    if any(v not in (0, 1) for v in relevance) or not 0 < sum(relevance) < len(docs):
        raise ValueError('At least one positive and one negative are required')
    selected = record['selected_positive_id']
    if selected not in ids or relevance[ids.index(selected)] != 1:
        raise ValueError('Selected ID disagrees with relevance')
    original = set(record['original_relevant_docids'])
    if selected not in original or any(i in original for i, r in zip(ids, relevance) if r == 0):
        raise ValueError('A known positive was relabeled negative')
    if sorted(record['teacher_ranking']) != list(range(1, len(docs) + 1)):
        raise ValueError('Teacher ranking must remain a separate valid permutation')


def simplify_record(record):
    """Expose all positives; the seeded in-batch representative comes first."""
    validate_record(record)
    pos = record['document_ids'].index(record['selected_positive_id'])
    positives = [pos] + [i for i, r in enumerate(record['relevance']) if r and i != pos]
    negatives = [i for i, r in enumerate(record['relevance']) if not r]
    order = positives + negatives
    public = dict(id=record['record_id'], query=record['query'], positives=[record['document'][i] for i in positives],
                  negatives=[record['document'][i] for i in negatives], source=record['source'])
    metadata = {k: v for k, v in record.items() if k not in {'query', 'document', 'relevance', 'source'}}
    metadata['document_ids'] = [record['document_ids'][i] for i in order]
    positions = {old+1: new+1 for new, old in enumerate(order)}
    metadata['teacher_ranking'] = [positions[i] for i in record['teacher_ranking']]
    return public, metadata


def _join_preprocessing_metadata(record, metadata):
    """Adapt the public interface to internal candidate/label tensors automatically."""
    positives, negatives = record.get('positives'), record.get('negatives')
    if not isinstance(positives, list) or not positives or not all(isinstance(x, str) and x.strip() for x in positives):
        raise ValueError('positives must contain nonempty texts; regenerate legacy data from raw inputs')
    if not isinstance(negatives, list) or not negatives or not all(isinstance(x, str) and x.strip() for x in negatives):
        raise ValueError('negatives must contain nonempty texts')
    if not isinstance(record.get('query'), str) or not record['query'].strip():
        raise ValueError('query must be nonempty text')
    normalized = dict(query=record['query'], document=[*positives, *negatives],
                      relevance=[1]*len(positives)+[0]*len(negatives), source=record.get('source', 'unknown'))
    if metadata['record_id'] != record['id']:
        raise ValueError('Training record does not match its metadata sidecar')
    normalized.update(metadata)
    validate_record(normalized)
    return normalized


READY_SCHEMA = 'embedding_candidates_v2'


def document_key(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def prepare_training_record(public, metadata):
    """Compile static sample semantics once, keeping audit fields out of training."""
    record = _join_preprocessing_metadata(public, metadata)
    n = len(record['document'])
    grades, ranks = [0] * n, [0] * n
    for position, candidate in enumerate(record['teacher_ranking']):
        grades[candidate - 1] = 3 if position == 0 else 2 if position < 5 else 1 if position < 10 else 0
        ranks[candidate - 1] = n - position
    return dict(schema=READY_SCHEMA, id=public['id'], query=record['query'], source=record['source'],
                document=record['document'], document_ids=record['document_ids'],
                relevance=record['relevance'], graded_relevance=grades, rank_labels=ranks,
                document_keys=[document_key(d) for d in record['document']],
                known_document_ids=sorted(set(record['original_relevant_docids']) | set(record['document_ids'])))


def compile_directory(directory):
    """Build self-contained training files from public records and audit sidecars."""
    directory = Path(directory)
    outputs = []
    for split in ('train', 'dev'):
        source = directory / f'{split}.jsonl'
        if not source.exists():
            continue
        output = directory / f'{split}.ready.jsonl'
        if output.exists():
            raise FileExistsError(f'Prepared training file already exists: {output}')
        with (directory / f'{split}.metadata.jsonl').open() as f:
            metadata = {}
            for line in f:
                item = json.loads(line)
                if item['record_id'] in metadata:
                    raise ValueError('Duplicate metadata record ID')
                metadata[item['record_id']] = item
        temp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False) as f:
                temp = Path(f.name)
                seen = set()
                with source.open() as input_file:
                    for line in input_file:
                        record = json.loads(line)
                        if record['id'] in seen:
                            raise ValueError('Duplicate public record ID')
                        seen.add(record['id'])
                        prepared = prepare_training_record(record, metadata[record['id']])
                        f.write(json.dumps(prepared, ensure_ascii=False) + '\n')
                if seen != set(metadata):
                    raise ValueError('Training rows and metadata IDs must match')
                f.flush()
                os.fsync(f.fileno())
            temp.replace(output)
            outputs.append(output)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    return outputs


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def query_key(text):
    # Matches the audit's conservative punctuation-normalized duplicate screen.
    return ' '.join(re.findall(r'\w+', normalize(text)))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def convert_multi_positive(row, row_index, seed=42):
    query, docs = parse_query_and_passages(_user_prompt(row['prompt']))
    ids = row['initial_list']
    if len(ids) != len(docs) or len(set(ids)) != len(ids):
        raise ValueError('Invalid candidate ID/text alignment')
    relevant = set(row['relevant_docids'])
    if not relevant.issubset(ids):
        raise ValueError('Relevant IDs outside candidates')
    ranking = parse_teacher_ranking(row['label'], len(ids))
    if row['final_list'] != [ids[i - 1] for i in ranking]:
        raise ValueError('Teacher permutation disagrees with final_list')
    by_text = defaultdict(list)
    for i, doc in enumerate(docs):
        by_text[normalize(doc)].append(i)
    if any(len({ids[i] in relevant for i in group}) > 1 for group in by_text.values()):
        return None, 'conflicting_relevance'
    # Keep first retrieval occurrence of identical text, never duplicate its supervision.
    unique = sorted(group[0] for group in by_text.values())
    positives = sorted((i for i in unique if ids[i] in relevant), key=lambda i: ids[i])
    negatives = [i for i in unique if ids[i] not in relevant]
    if not positives:
        return None, 'no_positive'
    if not negatives:
        return None, 'no_negative'
    # Per-query RNG makes choice independent of row order, exclusions and split assignment.
    entropy = hashlib.sha256(f'{seed}\0{query_key(query)}'.encode()).digest()
    selected = random.Random(int.from_bytes(entropy, 'big')).choice(positives)
    kept = unique
    new_position = {old: position + 1 for position, old in enumerate(kept)}
    record = dict(record_id=f'reasonrank/train/{row_index}', raw_row=row_index,
        source=row['dataset'], query=query, document=[docs[i] for i in kept],
        document_ids=[ids[i] for i in kept], relevance=[int(ids[i] in relevant) for i in kept],
        selected_positive_id=ids[selected], original_relevant_docids=sorted(relevant),
        teacher_ranking=[new_position[i-1] for i in ranking if i-1 in new_position],
        positive_selection_seed=seed,
        removed_positive_ids=[],
        duplicate_texts_removed=len(ids)-len(unique))
    validate_record(record)
    return record, 'kept'


def split_records(records, pairs, dev_size, split_seed):
    """Stratified deterministic allocation; known similar pairs cannot cross splits."""
    if dev_size == 0:
        return list(records), []
    if not 0 < dev_size < len(records):
        raise ValueError('dev_size must be between zero and total records')
    parent = {r['raw_row']: r['raw_row'] for r in records}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for pair in pairs:
        if pair['a_split'] == pair['b_split'] == 'train':
            a, b = pair['a_row'], pair['b_row']
            if a in parent and b in parent:
                parent[find(b)] = find(a)
    groups = defaultdict(list)
    for r in records:
        groups[find(r['raw_row'])].append(r)
    by_source = defaultdict(list)
    for group in groups.values():
        # A cross-source same-problem group stays whole, assigned to a stable stratum.
        source = min(r['source'] for r in group)
        by_source[source].append(group)
    weights = {source: sum(map(len, gs)) for source, gs in by_source.items()}
    quotas = {source: dev_size * n // len(records) for source, n in weights.items()}
    remainder_order = sorted(weights, key=lambda s: (-(dev_size * weights[s] % len(records)), s))
    for source in remainder_order[:dev_size - sum(quotas.values())]:
        quotas[source] += 1
    selected = set()
    leftovers = []
    for source in sorted(by_source):
        gs = sorted(by_source[source], key=lambda g: min(r['raw_row'] for r in g))
        random.Random(f'{split_seed}:{source}').shuffle(gs)
        count = 0
        for group in gs:
            if count + len(group) <= quotas[source]:
                selected.update(r['raw_row'] for r in group)
                count += len(group)
            else:
                leftovers.append(group)
    # Fill exact remaining quota when possible without ever breaking a group.
    for group in sorted(leftovers, key=lambda g: (len(g), min(r['raw_row'] for r in g))):
        if len(selected) + len(group) <= dev_size:
            selected.update(r['raw_row'] for r in group)
    train = [r for r in records if r['raw_row'] not in selected]
    dev = [r for r in records if r['raw_row'] in selected]
    assert not {find(r['raw_row']) for r in train} & {find(r['raw_row']) for r in dev}
    return train, dev


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_jsonl(path, rows):
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/audit_reasonrank_bright',
                        help='Downloaded inputs and results/ from the BRIGHT audit')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/processed/reasonrank_multi')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=20260911)
    parser.add_argument('--dev-size', type=int, default=0)
    parser.add_argument('--include-msmarco', action='store_true')
    parser.add_argument('--compile-only', action='store_true', help='Compile existing public data and metadata for training')
    args = parser.parse_args()
    args.input = args.data_dir / "reasonrank/train.parquet"
    args.audit_dir = args.data_dir / "results"
    args.internal_matches = args.audit_dir / "internal_query_matches.json"
    if args.compile_only:
        for path in compile_directory(args.output_dir):
            print(path)
        return
    if args.output_dir.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {args.output_dir}')
    fingerprints = json.loads((args.data_dir/'download_manifest.json').read_text())
    expected = next(f for f in fingerprints['files'] if f['repo']=='liuwenhan/reasonrank_data_rl' and f['source_path']=='train.parquet')
    if sha(args.input) != expected['sha256']:
        raise ValueError('Input hash differs from audited parquet; exclusion row IDs are not portable')
    exclusions_path = args.audit_dir/'quarantine_manifest.json'
    exclusions = {r['row'] for r in json.loads(exclusions_path.read_text())['rows'] if r['split']=='train'}
    pairs = json.loads(args.internal_matches.read_text()) if args.dev_size else []
    records, decisions, seen = [], [], set()
    for row_index, row in enumerate(iter_parquet_rows(args.input)):
        record = None
        if row_index in exclusions:
            reason = 'bright_quarantine'
        elif row['dataset'] == 'msmarco' and not args.include_msmarco:
            reason = 'excluded_source'
        else:
            record, reason = convert_multi_positive(row, row_index, args.seed)
            if record:
                key = query_key(record['query'])
                if key in seen:
                    record, reason = None, 'duplicate_query'
                else:
                    seen.add(key)
        decisions.append(dict(raw_row=row_index, source=row['dataset'], decision=reason))
        if record:
            records.append(record)
    train, dev = split_records(records, pairs, args.dev_size, args.split_seed)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.reasonrank-', dir=args.output_dir.parent))
    try:
        for split_name, split_rows in [('train', train), ('dev', dev)]:
            pairs = [simplify_record(r) for r in split_rows]
            write_jsonl(stage/f'{split_name}.jsonl', [p[0] for p in pairs])
            write_jsonl(stage/f'{split_name}.metadata.jsonl', [p[1] for p in pairs])
        compile_directory(stage)
        write_jsonl(stage/'decisions.jsonl', decisions)
        split_audit = dict(train_rows=[r['raw_row'] for r in train], dev_rows=[r['raw_row'] for r in dev],
            normalized_query_overlap=0, known_similar_group_overlap=0,
            note='Uses audited cosine>=0.90 groups; not exhaustive semantic deduplication.')
        write_json(stage/'split_audit.json', split_audit)
        stats = dict(input_rows=len(decisions), kept=len(records), train=len(train), dev=len(dev),
            requested_dev=args.dev_size, positive_selection_seed=args.seed, split_seed=args.split_seed,
            decisions=dict(Counter(d['decision'] for d in decisions)),
            candidate_counts=dict(sorted(Counter(len(r['document']) for r in records).items())),
            train_sources=dict(Counter(r['source'] for r in train)), dev_sources=dict(Counter(r['source'] for r in dev)),
            removed_positive_ids=sum(len(r['removed_positive_ids']) for r in records),
            positive_counts=dict(sorted(Counter(sum(r['relevance']) for r in records).items())))
        write_json(stage/'summary.json', stats)
        artifacts = [dict(role=role,path=str((args.output_dir/name).resolve()),sha256=sha(stage/name))
                     for role,name in [('train','train.ready.jsonl'),('dev','dev.ready.jsonl'),('public_train','train.jsonl'),('public_dev','dev.jsonl'),('train_metadata','train.metadata.jsonl'),('dev_metadata','dev.metadata.jsonl'),('decisions','decisions.jsonl'),('split_audit','split_audit.json')]]
        write_json(stage/'manifest.json', dict(version=1,split_seed=args.split_seed,
            positive_selection_seed=args.seed, selection='fixed_budget_final_checkpoint' if not dev else 'independent_dev', input=dict(path=str(args.input.resolve()),sha256=sha(args.input),revision=expected['revision']),
            exclusions=dict(path=str(exclusions_path.resolve()),sha256=sha(exclusions_path)),
            internal_matches=dict(path=str(args.internal_matches.resolve()),sha256=sha(args.internal_matches)) if args.dev_size else None,
            artifacts=artifacts,split_audit=artifacts[-1],evaluation_protocol=None,
            note='Prepared candidate data; padding and batch masks are handled by the trainer.'))
        stage.rename(args.output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
