"""nnU-Net v2.8.1 trainer plug-ins for controlled augmentation ablations.

nnU-Net discovers trainer classes by name on its package search path. Install this
project editable and pass ``-tr <class name>``. The imports are intentionally guarded
so non-training utilities remain usable on machines without nnU-Net/Torch.
"""
from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
    from mamaguardnet.augmentation.tensor_affine import apply_affine
    from scipy.ndimage import zoom
    from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
    from batchgeneratorsv2.transforms.utils.random import RandomTransform
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoDA import nnUNetTrainerNoDA
except ImportError as error:  # pragma: no cover - exercised only in training environment
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None


if _IMPORT_ERROR is None:
    class _SeedMixin:
        def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
            super().__init__(plans, configuration, fold, dataset_json, device)
            self.save_every = int(os.getenv("MAMAGUARD_SAVE_EVERY", "10"))
            seed = int(os.getenv("MAMAGUARD_SEED", "20261031")) + int(self.local_rank)
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)


    class SharedGlobalAffineTransform(ImageOnlyTransform):
        """One positive gain and offset shared by all phases and voxels."""

        def __init__(self, gain_log_std: float = 0.12, offset_std: float = 0.05):
            super().__init__()
            self.gain_log_std = gain_log_std
            self.offset_std = offset_std

        def get_parameters(self, **data_dict) -> dict:
            image = data_dict["image"]
            if image.shape[0] != 3:
                raise ValueError(f"MAMAGuard expects exactly 3 ordered DCE channels, got {image.shape[0]}")
            return {
                "gain": float(np.exp(np.random.normal(0, self.gain_log_std))),
                "offset": float(np.random.normal(0, self.offset_std)),
            }

        def _apply_to_image(self, image: torch.Tensor, **params) -> torch.Tensor:
            return apply_affine(image, params["gain"], params["offset"])


    class SharedAffineFieldTransform(ImageOnlyTransform):
        """Apply one smooth positive gain/offset field to every DCE phase."""

        def __init__(self, gain_log_std: float = 0.12, offset_std: float = 0.05, grid: int = 4):
            super().__init__()
            self.gain_log_std = gain_log_std
            self.offset_std = offset_std
            self.grid = grid

        def get_parameters(self, **data_dict) -> dict:
            image = data_dict["image"]
            if image.shape[0] != 3:
                raise ValueError(f"MAMAGuard expects exactly 3 ordered DCE channels, got {image.shape[0]}")
            spatial_dims = image.ndim - 1
            low_shape = (self.grid,) * spatial_dims
            target = tuple(image.shape[1:])
            factors = tuple(size / self.grid for size in target)
            gain = zoom(np.random.normal(0, self.gain_log_std, low_shape), factors, order=1)
            offset = zoom(np.random.normal(0, self.offset_std, low_shape), factors, order=1)
            slices = tuple(slice(0, size) for size in target)
            return {"gain": np.exp(gain[slices])[None], "offset": offset[slices][None]}

        def _apply_to_image(self, image: torch.Tensor, **params) -> torch.Tensor:
            return apply_affine(image, params["gain"], params["offset"])


    class IndependentAffineTransform(ImageOnlyTransform):
        """Deliberately phase-independent intensity ablation."""

        def __init__(self, gain_log_std: float = 0.12, offset_std: float = 0.05):
            super().__init__()
            self.gain_log_std = gain_log_std
            self.offset_std = offset_std

        def get_parameters(self, **data_dict) -> dict:
            image = data_dict["image"]
            shape = (image.shape[0],) + (1,) * (image.ndim - 1)
            return {
                "gain": np.exp(np.random.normal(0, self.gain_log_std, shape)),
                "offset": np.random.normal(0, self.offset_std, shape),
            }

        def _apply_to_image(self, image: torch.Tensor, **params) -> torch.Tensor:
            return apply_affine(image, params["gain"], params["offset"])


    def _append_transform(base, transform):
        # ComposeTransforms is the current nnU-Net v2.8.1 return type.
        if not hasattr(base, "transforms"):
            raise TypeError("Unsupported nnU-Net transform pipeline; expected ComposeTransforms")
        base.transforms.append(RandomTransform(transform, apply_probability=float(os.getenv("MAMAGUARD_AUG_P", "0.5"))))
        return base


    def _strip_default_intensity(base):
        """Retain nnU-Net spatial/label transforms but remove its intensity policies."""
        blocked = {
            "GaussianNoiseTransform",
            "GaussianBlurTransform",
            "MultiplicativeBrightnessTransform",
            "ContrastTransform",
            "SimulateLowResolutionTransform",
            "GammaTransform",
        }
        kept = []
        for transform in base.transforms:
            wrapped = getattr(transform, "transform", transform)
            if type(wrapped).__name__ not in blocked:
                kept.append(transform)
        base.transforms = kept
        return base


    class nnUNetTrainerMAMAGuardNoDA(_SeedMixin, nnUNetTrainerNoDA):
        pass


    class nnUNetTrainerMAMAGuardStandard(_SeedMixin, nnUNetTrainer):
        pass


    class nnUNetTrainerMAMAGuardIndependent(_SeedMixin, nnUNetTrainer):
        @staticmethod
        def get_training_transforms(*args, **kwargs):
            base = _strip_default_intensity(nnUNetTrainer.get_training_transforms(*args, **kwargs))
            return _append_transform(base, IndependentAffineTransform())


    class nnUNetTrainerMAMAGuardShared(_SeedMixin, nnUNetTrainer):
        @staticmethod
        def get_training_transforms(*args, **kwargs):
            base = _strip_default_intensity(nnUNetTrainer.get_training_transforms(*args, **kwargs))
            return _append_transform(base, SharedGlobalAffineTransform())


    class nnUNetTrainerMAMAGuardKinetics(_SeedMixin, nnUNetTrainer):
        """Primary method: positive shared voxelwise affine DCE augmentation.

        For phases t and s, I'_t-I'_s = a(x)(I_t-I_s), where a(x)>0.
        Therefore signs and pairwise ordering are exactly preserved. This does not
        claim preservation of difference magnitude, percentage enhancement, or
        invariance under separately sampled noise/nonlinear transforms.
        """

        @staticmethod
        def get_training_transforms(*args, **kwargs):
            transform = SharedAffineFieldTransform(
                gain_log_std=float(os.getenv("MAMAGUARD_GAIN_LOG_STD", "0.12")),
                offset_std=float(os.getenv("MAMAGUARD_OFFSET_STD", "0.05")),
                grid=int(os.getenv("MAMAGUARD_FIELD_GRID", "4")),
            )
            base = _strip_default_intensity(nnUNetTrainer.get_training_transforms(*args, **kwargs))
            return _append_transform(base, transform)


    class nnUNetTrainerMAMAGuardSmoke(nnUNetTrainerMAMAGuardKinetics):
        """Bounded integration test for data, augmentation, training, and validation."""

        def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
            super().__init__(plans, configuration, fold, dataset_json, device)
            self.num_epochs = int(os.getenv("MAMAGUARD_SMOKE_EPOCHS", "2"))
            self.num_iterations_per_epoch = int(os.getenv("MAMAGUARD_SMOKE_TRAIN_ITERS", "5"))
            self.num_val_iterations_per_epoch = int(os.getenv("MAMAGUARD_SMOKE_VAL_ITERS", "2"))

