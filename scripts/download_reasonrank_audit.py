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


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def download(data_dir):
    data_dir = Path(data_dir)
    manifest_path = data_dir / 'download_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files': []}
    recorded = {(f['repo'], f['revision'], f['source_path']): f for f in previous['files']}
    files = []
    for prefix, (repo, revision, paths) in SOURCES.items():
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
            manifest = dict(repositories={r: rev for r, rev, _ in SOURCES.values()},
                            files=list(recorded.values()))
            temporary_manifest = manifest_path.with_suffix('.json.tmp')
            temporary_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
            temporary_manifest.replace(manifest_path)
            print(f'Ready: {prefix}/{source}')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/audit_reasonrank_bright')
    args = parser.parse_args()
    download(args.data_dir)


if __name__ == '__main__':
    main()
