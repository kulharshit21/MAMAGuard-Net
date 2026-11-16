"""Create an isolated Git repository containing code, configs and research only."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = a.destination.resolve()
    if dest.exists():
        raise ValueError('Destination must not exist')
    dest.mkdir(parents=True)
    hashes = {}
    for folder in ['src', 'tests', 'configs', 'scripts', 'research', 'docs', 'registry',
                   'reports/scientific_audit', 'reports/generated/secondary_validation']:
        for path in (root/folder).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ('.py', '.sh', '.yaml', '.json', '.md', '.csv', '.bib', '.log', '.txt'):
                continue
            relative = path.relative_to(root)
            target = dest/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    for name in ('pyproject.toml','README.md','.gitignore','DECISIONS.md','EXPERIMENT_LOG.md','PROJECT_STATUS.md'):
        shutil.copy2(root/name, dest/name)
        hashes[name] = hashlib.sha256((dest/name).read_bytes()).hexdigest()
    (dest/'SOURCE_HASHES.json').write_text(json.dumps(hashes, indent=2)+'\n')
    subprocess.run(['git', 'init', str(dest)], check=True)
    subprocess.run(['git', '-C', str(dest), 'config', 'core.autocrlf', 'false'], check=True)
    subprocess.run(['git', '-C', str(dest), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(dest), '-c', 'user.name=MAMAGuard Research',
                    '-c', 'user.email=research@localhost', 'commit', '-m',
                    'Stage unapproved secondary research implementation'], check=True)
    subprocess.run(['git', '-C', str(dest), 'rev-parse', 'HEAD'], check=True)


if __name__ == '__main__':
    main()
