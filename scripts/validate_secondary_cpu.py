"""Run isolated CPU validation only; never launches training or changes approvals."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True,
                        help='Read-only completed A3 directory containing plans.json and dataset.json')
    args = parser.parse_args()
    if Path(sys.prefix).name != '.venv-secondary':
        raise RuntimeError('Requires a new isolated .venv-secondary environment')
    root = Path(__file__).resolve().parents[1]
    for name in ('plans.json','dataset.json'):
        if not (args.model_root/name).is_file():
            raise FileNotFoundError(args.model_root/name)
    import psutil
    if psutil.virtual_memory().available < 12 * 1024**3:
        raise RuntimeError('CPU validation deferred: requires 12 GiB available RAM')
    if shutil.disk_usage(root).free < 12 * 1024**3:
        raise RuntimeError('CPU validation deferred: requires 12 GiB temporary disk capacity')
    if hasattr(os, 'getloadavg') and os.getloadavg()[1] > .8 * (os.cpu_count() or 1):
        raise RuntimeError('CPU validation deferred: five-minute system load exceeds 80% of logical CPUs')
    if hasattr(os, 'nice'):
        os.nice(15)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', nnUNet_compile='false',
               PYTHONPATH=str(root/'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               MAMAGUARD_INTEGRATION_MODEL_ROOT=str(args.model_root.resolve()))
    evidence = root/'reports/generated/secondary_cpu_validation'
    evidence.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, 'scripts/install_secondary_plugin.py'],
        [sys.executable, '-m', 'pytest', '-ra', '-p', 'no:cacheprovider'],
    ]
    for index, command in enumerate(commands):
        print('Executing:', command, flush=True)
        with (evidence/f'step_{index}.log').open('w') as log:
            result = subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        print((evidence/f'step_{index}.log').read_text(), flush=True)
        if result.returncode:
            return result.returncode
    print('CPU validation commands passed; secondary approval remains unchanged.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
