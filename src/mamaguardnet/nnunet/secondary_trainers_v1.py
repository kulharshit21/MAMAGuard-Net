"""Isolated candidate trainers. Never installed into the active campaign automatically."""
import json
import os

import numpy as np

from mamaguardnet.augmentation.adaptive_v1 import adaptive_affine
from mamaguardnet.augmentation.smooth_control import sample_smooth_affine_fields
from mamaguardnet.nnunet import trainers as _base

if _base._IMPORT_ERROR is not None:
    raise ImportError('Secondary trainers require the complete isolated nnUNet dependency set') from _base._IMPORT_ERROR

from mamaguardnet.nnunet.trainers import (
    ImageOnlyTransform,
    _append_transform,
    _SeedMixin,
    _strip_default_intensity,
    apply_affine,
    nnUNetTrainer,
    torch,
)


def parameters():
    return json.loads(os.environ.get('MAMAGUARD_SECONDARY_PARAMETERS', '{}'))


class IndependentSmoothV1(ImageOnlyTransform):
    def __init__(self, **params):
        super().__init__()
        self.params = params

    def get_parameters(self, **data_dict):
        image = data_dict['image']
        if image.shape[0] != 3:
            raise ValueError('Expected pre, early-post, later-post channels')
        rng = np.random.default_rng(np.random.randint(0, 2**32-1))
        fields = sample_smooth_affine_fields(tuple(image.shape[1:]), rng=rng,
                                            shared_across_phases=False, **self.params)
        return {'gain': fields.gain, 'offset': fields.offset}

    def _apply_to_image(self, image, **params):
        return apply_affine(image, params['gain'], params['offset'])


class AdaptiveV1(ImageOnlyTransform):
    def __init__(self, mode='adaptive', **params):
        super().__init__()
        self.mode = mode
        self.params = params

    def _apply_to_image(self, image, **params):
        if image.shape[0] != 3:
            raise ValueError('Expected three ordered channels')
        rng = np.random.default_rng(np.random.randint(0, 2**32-1))
        result = adaptive_affine(image.detach().cpu().numpy(), rng=rng, mode=self.mode, **self.params)
        return torch.as_tensor(result.image, dtype=image.dtype, device=image.device)


class nnUNetTrainerMAMAGuardA5V1(_SeedMixin, nnUNetTrainer):
    @staticmethod
    def get_training_transforms(*args, **kwargs):
        base = _strip_default_intensity(nnUNetTrainer.get_training_transforms(*args, **kwargs))
        return _append_transform(base, IndependentSmoothV1(**parameters()))


class nnUNetTrainerMAMAGuardA6V1(_SeedMixin, nnUNetTrainer):
    @staticmethod
    def get_training_transforms(*args, **kwargs):
        base = _strip_default_intensity(nnUNetTrainer.get_training_transforms(*args, **kwargs))
        return _append_transform(base, AdaptiveV1(**parameters()))


class nnUNetTrainerMAMAGuardA7V1(_SeedMixin, nnUNetTrainer):
    @staticmethod
    def get_training_transforms(*args, **kwargs):
        base = _strip_default_intensity(nnUNetTrainer.get_training_transforms(*args, **kwargs))
        return _append_transform(base, AdaptiveV1('shifted', **parameters()))
