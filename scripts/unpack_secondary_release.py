"""Verify and extract a release into a new directory, without installing or launching."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def unpack(manifest_path, destination):
    manifest = json.loads(manifest_path.read_text())
    name = manifest['archive']
    if Path(name).name != name:
        raise ValueError('Archive name must be a basename')
    archive = manifest_path.parent/name
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise ValueError('Archive hash mismatch')
    if destination.exists():
        raise ValueError('Destination must not exist')
    with tarfile.open(archive,'r:gz') as source:
        members = source.getmembers()
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or '\\' in member.name or ':' in member.name or not member.isfile() or member.name in names:
                raise ValueError('Unsafe archive entry: '+member.name)
            names.add(member.name)
        if len(members) != manifest['file_count']:
            raise ValueError('File-count mismatch')
        destination.mkdir(parents=True)
        # All entries are validated regular files, no links or parent components.
        for member in members:
            path = destination/member.name
            path.parent.mkdir(parents=True,exist_ok=True)
            with source.extractfile(member) as data, path.open('xb') as output:
                output.write(data.read())
    hashes = json.loads((destination/'SOURCE_HASHES.json').read_text())
    for name,digest in hashes.items():
        path = destination/name
        if destination.resolve() not in path.resolve().parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Extracted source mismatch: '+name)
    print('VERIFIED EXTRACTION:',destination,'commit=',manifest['git_commit'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    a = p.parse_args()
    unpack(a.manifest,a.destination)
