# MAMAGuard-Net

Research code for a five-fold comparison of augmentation strategies in three-phase breast DCE-MRI segmentation with nnU-Net v2 ResEncL. The internal A0–A4 campaign used seed 20261031. This repository is for research reproducibility.

| Configuration | Training augmentation |
|---|---|
| A0 | No augmentation |
| A1 | Standard nnU-Net augmentation |
| A2 | Independent phase intensity perturbation |
| A3 | Shared affine transform |
| A4 | Phase ordering constrained augmentation |

## Repository map

- src/mamaguardnet: dataset, training, augmentation and evaluation implementation.
- configs/experiments: A0–A4 settings. Candidate configs are development work.
- scripts: manifest construction, split generation, trainer installation, evaluation and paired comparisons.
- cluster: DGX launch and recovery examples. Review paths and GPU assignments before use.
- requirements: DGX dependency record. tests: source validation suite.

## Reproduction

1. Obtain MAMA-MIA from https://doi.org/10.7303/syn60868042 under the original repository terms. Use the documented precontrast, early postcontrast and later postcontrast channel order. MRI files and patient-level artifacts are held outside this repository.
2. Create a Python 3.12 environment and run `python -m pip install -e ".[training,analysis]"`. Review requirements/dgx-cu130.txt when reproducing the DGX dependency environment.
3. Set nnUNet_raw, nnUNet_preprocessed and nnUNet_results to your own paths. Consult `python scripts/stage_nnunet_mamamia.py --help`, `python scripts/create_protocol_splits.py --help` and `python scripts/install_nnunet_plugin.py --help` before staging and installing the trainer. Use the archived split for exact fold membership when access permits.
4. Select the corresponding YAML in configs/experiments and consult `python -m mamaguardnet.cli train --help`. Evaluate held-out predictions using `python scripts/evaluate_predictions.py --help`. See `python scripts/compare_primary_physical.py --help` for the paired analysis.

## Checkpoints and access

The 25 final checkpoints, nnU-Net plans, dataset descriptors and SHA-256 manifest are pinned at https://huggingface.co/harshitrk07/MAMAGuard-Net/tree/4e8d231c3ad821be90be8b5c30b329a658c26d59. Choose the experiment and fold needed. Use checkpoint_final.pth with the matching model directory and custom trainer. The checkpoints represent internal evaluation only.

This private source snapshot grants access to invited collaborators. No reusable code or weights licence has been declared. Third-party software and the source dataset retain their respective terms. Participant-level manifests, predictions and MRI files are excluded. The independent Yunnan cohort was not used in the manuscript evaluation.

The corresponding author is Harshit Ravindra Kulkarni. Please cite the manuscript and the pinned artifact revision when referring to this work.
