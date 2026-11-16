"""Archive a clean isolated package, with byte hashes and Git provenance."""
import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if subprocess.check_output(['git','-C',str(a.package),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Package must have a clean Git tree')
    commit = subprocess.check_output(['git','-C',str(a.package),'rev-parse','HEAD'],text=True).strip()
    hashes = json.loads((a.package/'SOURCE_HASHES.json').read_text())
    for name,digest in hashes.items():
        if hashlib.sha256((a.package/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Package hash mismatch: '+name)
    a.output.mkdir(parents=True,exist_ok=False)
    archive = a.output/'secondary_readiness.tar.gz'
    files = sorted([a.package/name for name in hashes] + [a.package/'SOURCE_HASHES.json'])
    with tarfile.open(archive,'w:gz') as target:
        for path in files:
            if path.is_symlink():
                raise RuntimeError('Symlinks not allowed')
            target.add(path,arcname=path.relative_to(a.package).as_posix(),recursive=False)
    manifest = {'archive':archive.name,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                'git_commit':commit,'file_count':len(files),'status':'UNAPPROVED_CPU_VALIDATION_ONLY'}
    (a.output/'secondary_readiness.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
