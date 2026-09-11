"""Download the ReasonRank/BRIGHT audit inputs and record local fingerprints."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'reasonrank': ('liuwenhan/reasonrank_data_rl', '28c5836408857149b80dc352ed362eadc1199a50',
                  ['train.parquet', 'val.parquet']),
    'bright': ('xlangai/BRIGHT', '3066d29c9651a576c8aba4832d249807b181ecae',
               [f'examples/{domain}-00000-of-00001.parquet' for domain in (
                   'aops', 'biology', 'earth_science', 'economics', 'leetcode', 'pony',
                   'psychology', 'robotics', 'stackoverflow', 'sustainable_living',
                   'theoremqa_questions', 'theoremqa_theorems')]),
}
REASONRANK_DOCUMENT_SOURCE = (
    'liuwenhan/reasonrank_data_13k',
    '09c3ac0f8dee374207592e118866d9bf943e74cc',
    [f'id_doc/{source}.json' for source in (
        'biology', 'earth_science', 'economics', 'leetcode', 'math-qa',
        'math-theorem', 'robotics', 'stackoverflow', 'sustainable_living',
    )],
)
BRIGHT_DOCUMENT_PATHS = [
    f'documents/{domain}-00000-of-00001.parquet' for domain in (
        'aops', 'biology', 'earth_science', 'economics', 'leetcode', 'pony',
        'psychology', 'robotics', 'stackoverflow', 'sustainable_living',
        'theoremqa_questions', 'theoremqa_theorems',
    )
]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def download(data_dir, include_bright_documents=False, include_reasonrank_documents=False):
    data_dir = Path(data_dir)
    manifest_path = data_dir / 'download_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files': []}
    recorded = {(f['repo'], f['revision'], f['source_path']): f for f in previous['files']}
    files = []
    sources = dict(SOURCES)
    if include_bright_documents:
        repo, revision, paths = sources['bright']
        sources['bright'] = (repo, revision, [*paths, *BRIGHT_DOCUMENT_PATHS])
    if include_reasonrank_documents:
        sources['reasonrank_documents'] = REASONRANK_DOCUMENT_SOURCE
    for prefix, (repo, revision, paths) in sources.items():
        for source in paths:
            path = data_dir / prefix / source
            path.parent.mkdir(parents=True, exist_ok=True)
            # Only reuse files whose provenance was recorded by an earlier download.
            known = recorded.get((repo, revision, source))
            if path.exists():
                if known is None or digest(path) != known['sha256']:
                    raise ValueError(f'Existing file has no matching download record: {path}')
            else:
                temporary = None
                try:
                    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{source}'
                    with urllib.request.urlopen(url, timeout=180) as response:
                        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                            temporary = Path(output.name)
                            shutil.copyfileobj(response, output)
                    if known and digest(temporary) != known['sha256']:
                        raise ValueError(f'Download differs from recorded input: {path}')
                    temporary.replace(path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            files.append(dict(repo=repo, revision=revision, source_path=source,
                              bytes=path.stat().st_size, sha256=digest(path)))
            # Save progress so an interrupted download can resume.
            recorded[(repo, revision, source)] = files[-1]
            manifest = dict(repositories={r: rev for r, rev, _ in sources.values()},
                            files=list(recorded.values()))
            temporary_manifest = manifest_path.with_suffix('.json.tmp')
            temporary_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
            temporary_manifest.replace(manifest_path)
            print(f'Ready: {prefix}/{source}')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/audit_reasonrank_bright')
    parser.add_argument(
        '--include-bright-documents',
        action='store_true',
        help='Also download the official BRIGHT documents configuration',
    )
    parser.add_argument(
        '--include-reasonrank-documents',
        action='store_true',
        help='Also download the canonical ReasonRank id_doc mappings used by G1 indexes',
    )
    args = parser.parse_args()
    download(
        args.data_dir,
        include_bright_documents=args.include_bright_documents,
        include_reasonrank_documents=args.include_reasonrank_documents,
    )


if __name__ == '__main__':
    main()
