"""Download the pinned audit inputs and verify their recorded SHA-256 hashes."""
from pathlib import Path
import hashlib,json,urllib.request
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'paper/audits/reasonrank_bright/download_manifest.json').read_text())
for f in manifest['files']:
 prefix='reasonrank' if f['repo']=='liuwenhan/reasonrank_data_rl' else 'bright'
 path=ROOT/'data/audit_reasonrank_bright'/prefix/f['source_path']
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  url=f"https://huggingface.co/datasets/{f['repo']}/resolve/{f['revision']}/{f['source_path']}"
  with urllib.request.urlopen(url,timeout=180) as response: payload=response.read()
  if hashlib.sha256(payload).hexdigest()!=f['sha256']:raise ValueError(f'Hash mismatch: {path}')
  path.write_bytes(payload)
 if hashlib.sha256(path.read_bytes()).hexdigest()!=f['sha256']:raise ValueError(f'Hash mismatch: {path}')
 print(f'Verified {prefix}/{f["source_path"]}')
