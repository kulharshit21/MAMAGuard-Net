"""Explicit installation into a NEW isolated environment only, never active DGX .venv."""
import sys
from pathlib import Path


def main():
    prefix = Path(sys.prefix).resolve()
    if prefix.name != '.venv-secondary':
        raise RuntimeError('Requires a dedicated .venv-secondary environment')
    import nnunetv2
    target = Path(nnunetv2.__file__).resolve().parent/'training/nnUNetTrainer/variants/mamaguard_secondary_v1.py'
    content = ('from mamaguardnet.nnunet.secondary_trainers_v1 import (\n'
                      '    nnUNetTrainerMAMAGuardA5V1, nnUNetTrainerMAMAGuardA6V1,\n'
                      '    nnUNetTrainerMAMAGuardA7V1,\n)\n')
    if prefix not in target.parents:
        raise RuntimeError('Refusing non-isolated target')
    if target.exists() and target.read_text() != content:
        raise RuntimeError('Refusing overwrite of a different plugin')
    if not target.exists():
        target.write_text(content)
    print(target)


if __name__ == '__main__':
    main()
