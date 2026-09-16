#!/usr/bin/env python3
"""Freeze ReasonEmbed v0928 candidates for the existing G2 loader (binary labels).

On the training machine, download the raw files, then run this script:

  hf download hanhainebula/reason-embed-data --repo-type dataset \
    --revision 8dada2e649a21913df303510980ebb8ed1de541e \
    --include 'reason-embed-data-0928/*.jsonl' --local-dir data/raw/reasonembed
  python scripts/prepare_reasonembed.py --input-dir data/raw/reasonembed \
    --output-dir data/processed/reasonembed_g2

Output train.jsonl uses embedding_candidates_v2, directly readable by G2.
Default: one seeded representative positive + up to 15 seeded negatives, matching
G2's 16-document candidate budget. ALL known positives remain in filtering keys.
--all-positives retains every positive and adds up to slate_size-1 negatives;
this produces variable, potentially much larger slates. No runtime resampling.

The released JSONL has pos/neg, not the paper's original 1--5 relevance scores.
relevance, graded_relevance and rank_labels therefore all encode binary 1/0 ties;
no teacher ordering is invented. Use relevance_scheme=binary for G2 CL.
Original query is used; reasoning_query is not a training input. Original prompt
is retained as original_prompt, but the existing G2 loader applies its own task
instruction (generic fallback for these domains), so this is not an exact
reproduction of the author's training recipe. No dev split or tail dropping here.

manifest.json records input/output hashes, selection and cleaning counts. Local
files' provenance is supplied by --revision; it cannot be verified from JSONL
alone. Output directories are never overwritten. Stdlib only; no GPU/API calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unicodedata

REPO = 'hanhainebula/reason-embed-data'
REVISION = '8dada2e649a21913df303510980ebb8ed1de541e'
DOMAINS = ('biology', 'earth_science', 'economics', 'psychology', 'robotics',
           'stackoverflow', 'sustainable_living', 'leetcode', 'pony', 'aops',
           'theoremqa_questions', 'theoremqa_theorems')


def text_key(text):
    normalized = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def unique_texts(values, field, stats):
    if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
        raise ValueError(f'{field} must be a list of strings')
    result = {}
    for text in values:
        if not text.strip():
            stats[f'empty_{field}'] += 1
            continue
        key = text_key(text)
        if key in result:
            stats[f'duplicate_{field}'] += 1
        else:
            result[key] = text
    return result


def convert_record(record, domain, line_number, seed=42, slate_size=16,
                   all_positives=False, stats=None):
    if stats is None:
        stats = Counter()
    if not isinstance(record, dict):
        raise ValueError('Each record must be an object')
    query = record.get('query')
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query must be a nonempty string')
    prompt = record.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('prompt must be a nonempty string')
    if slate_size < 2:
        raise ValueError('slate_size must be at least 2')
    positives = unique_texts(record.get('pos'), 'pos', stats)
    negatives = unique_texts(record.get('neg'), 'neg', stats)
    overlap = positives.keys() & negatives.keys()
    stats['positive_negative_overlap_removed'] += len(overlap)
    for key in overlap:
        del negatives[key]
    if not positives or not negatives:
        stats['dropped_no_positive' if not positives else 'dropped_no_negative'] += 1
        return None
    record_id = f'reasonembed/{domain}/{line_number}'
    # Independent per-row seeds make conversion invariant to domain selection/order.
    entropy = hashlib.sha256(f'{seed}/{record_id}/{text_key(query)}'.encode()).digest()
    rng = random.Random(int.from_bytes(entropy, 'big'))
    positive_keys = list(positives)
    representative = rng.choice(positive_keys)
    selected_pos = ([representative] + [k for k in positive_keys if k != representative]
                    if all_positives else [representative])
    selected_neg = rng.sample(list(negatives), min(slate_size - 1, len(negatives)))
    keys = selected_pos + selected_neg
    labels = [1] * len(selected_pos) + [0] * len(selected_neg)
    texts = [positives[k] for k in selected_pos] + [negatives[k] for k in selected_neg]
    stats['written'] += 1
    stats['known_positives'] += len(positive_keys)
    stats['selected_positives'] += len(selected_pos)
    stats['selected_negatives'] += len(selected_neg)
    stats['short_slates'] += int(len(selected_neg) < slate_size - 1)
    return dict(schema='embedding_candidates_v2', id=record_id, source=domain,
                query=query, original_prompt=prompt,
                document=texts, document_ids=[f'text:{k}' for k in keys],
                document_keys=keys, relevance=labels, graded_relevance=list(labels),
                rank_labels=list(labels),
                known_positive_keys=sorted(positive_keys),
                known_document_ids=sorted(f'text:{k}' for k in set(positive_keys) | set(keys)))


def input_files(directory, domains):
    if not directory.is_dir():
        raise ValueError(f'Missing input directory: {directory}')
    # Support both hf --local-dir layout and the dataset subdirectory itself.
    root = directory / 'reason-embed-data-0928'
    if not root.is_dir():
        root = directory
    paths = [(domain, root / f'{domain}-formatted.jsonl') for domain in domains]
    missing = [str(path) for _, path in paths if not path.is_file()]
    if missing:
        raise ValueError('Missing input files: ' + ', '.join(missing))
    return paths


def prepare(input_dir, output_dir, domains=DOMAINS, seed=42, slate_size=16,
            all_positives=False, revision=REVISION):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    domains = tuple(domains)
    if not domains or len(set(domains)) != len(domains) or set(domains) - set(DOMAINS):
        raise ValueError('Select distinct registered domains')
    if slate_size < 2:
        raise ValueError('slate_size must be at least 2')
    paths = input_files(input_dir, domains)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite output directory: {output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output_dir.name}-', dir=output_dir.parent))
    inputs, counts, seen_queries = [], {}, Counter()
    output_hash, total = hashlib.sha256(), 0
    try:
        with (staging / 'train.jsonl').open('wb') as output, (staging / 'decisions.jsonl').open('w') as decisions:
            for domain, path in paths:
                stats, input_hash = Counter(), hashlib.sha256()
                print(f'Processing {domain}: {path}', flush=True)
                with path.open('rb') as handle:
                    for line_number, raw in enumerate(handle, 1):
                        input_hash.update(raw)
                        if not raw.strip():
                            stats['blank_lines'] += 1
                            continue
                        stats['input_records'] += 1
                        try:
                            record = json.loads(raw)
                            converted = convert_record(record, domain, line_number, seed,
                                                       slate_size, all_positives, stats)
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f'{path}:{line_number}: {exc}') from exc
                        if converted is None:
                            decisions.write(json.dumps(dict(source=domain, line=line_number,
                                                            reason='no_usable_positive_or_negative')) + '\n')
                            continue
                        query_key = (domain, text_key(converted['query']))
                        stats['duplicate_queries_retained'] += int(seen_queries[query_key] > 0)
                        seen_queries[query_key] += 1
                        encoded = (json.dumps(converted, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
                        output.write(encoded)
                        output_hash.update(encoded)
                        total += 1
                        if stats['input_records'] % 1000 == 0:
                            print(f'  {domain}: read={stats["input_records"]}, written={stats["written"]}', flush=True)
                inputs.append(dict(source=domain, path=str(path), bytes=path.stat().st_size,
                                   sha256=input_hash.hexdigest()))
                counts[domain] = dict(stats)
        if not total:
            raise ValueError('No usable training records')
        manifest = dict(version=1, dataset=REPO, revision=revision, config='v0928',
                        input_provenance='local_files_revision_declared_not_verified',
                        schema='embedding_candidates_v2', labels='annotated_pos_neg_binary',
                        selection='fixed_seeded_candidates_no_dev', seed=seed,
                        positive_mode='all' if all_positives else 'representative',
                        slate_size=slate_size, max_negatives=slate_size - 1,
                        query_set='original', instruction='existing_G2_loader_task_prompt',
                        tail_dropping='deferred_to_training_loader', inputs=inputs, counts=counts,
                        artifacts=[dict(role='train', path='train.jsonl', records=total,
                                        bytes=(staging / 'train.jsonl').stat().st_size,
                                        sha256=output_hash.hexdigest())])
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if output_dir.exists():
            raise FileExistsError(f'Output directory appeared during conversion: {output_dir}')
        os.rename(staging, output_dir)
        print(f'Wrote {total} records: {output_dir / "train.jsonl"}\nSHA256: {output_hash.hexdigest()}', flush=True)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path, default=Path('data/raw/reasonembed'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed/reasonembed_g2'))
    parser.add_argument('--domains', nargs='+', choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--slate-size', type=int, default=16)
    parser.add_argument('--all-positives', action='store_true')
    parser.add_argument('--revision', default=REVISION, help='Declared revision of the downloaded raw files')
    args = parser.parse_args()
    prepare(args.input_dir, args.output_dir, args.domains, args.seed,
            args.slate_size, args.all_positives, args.revision)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
