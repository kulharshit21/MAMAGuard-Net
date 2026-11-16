"""CPU-only integration against actual mirrored ResEncL plans; no patient data."""
import importlib.util
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec('nnunetv2') is None,
                                reason='Requires isolated nnUNet environment')
ROOT = Path(__file__).resolve().parents[1]


def test_discovery_pipeline_loader_and_resencl_checkpoint(tmp_path, monkeypatch):
    import nnunetv2
    import torch
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    from mamaguardnet.nnunet import secondary_trainers_v1 as secondary
    torch.set_num_threads(1)
    monkeypatch.setenv('nnUNet_compile', 'false')
    monkeypatch.setenv('MAMAGUARD_AUG_P', '1')
    # nnU-Net 2.8.1 exposes path proxies that consult these variables dynamically.
    # Replacing the proxy objects with strings breaks their is_set() contract.
    monkeypatch.setenv('nnUNet_results', str(tmp_path/'results'))
    monkeypatch.setenv('nnUNet_preprocessed', str(tmp_path/'preprocessed'))
    configured_root = os.environ.get('MAMAGUARD_INTEGRATION_MODEL_ROOT')
    model_root = Path(configured_root) if configured_root else next((ROOT/'DGX_MIRROR/nnUNet_results/A3_shared_affine').glob('seed_*/Dataset501*/*/plans.json')).parent
    plans = json.loads((model_root/'plans.json').read_text())
    dataset = json.loads((model_root/'dataset.json').read_text())
    for name in ('nnUNetTrainerMAMAGuardA5V1', 'nnUNetTrainerMAMAGuardA6V1', 'nnUNetTrainerMAMAGuardA7V1'):
        cls = getattr(secondary, name)
        found = recursive_find_python_class(str(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer'), name,
                                            'nnunetv2.training.nnUNetTrainer')
        assert found is cls
        runtime_plans = deepcopy(plans)
        runtime_plans['continue_training'] = False
        trainer = cls(runtime_plans, '3d_fullres', 0, dataset, device=torch.device('cpu'))
        transforms = cls.get_training_transforms((8, 16, 16), (0., 0.), None, (0, 1, 2), False)
        names = [type(getattr(t, 'transform', t)).__name__ for t in transforms.transforms]
        assert 'GaussianNoiseTransform' not in names and 'SpatialTransform' in names
        data_dir = tmp_path/name
        data_dir.mkdir()
        image = np.random.default_rng(1).normal(size=(3, 8, 16, 16)).astype(np.float32)
        seg = np.zeros((1, 8, 16, 16), dtype=np.int16)
        seg[:, 2:6, 4:12, 4:12] = 1
        locations = np.argwhere(seg == 1)
        nnUNetDatasetNumpy.save_case(image, seg, {'class_locations': {1: locations}}, str(data_dir/'fixture'))
        loader = nnUNetDataLoader(nnUNetDatasetNumpy(str(data_dir)), 1, (8,16,16), (8,16,16),
                                 trainer.label_manager, transforms=transforms)
        batch = loader.generate_train_batch()
        assert tuple(batch['data'].shape) == (1,3,8,16,16)
        assert torch.isfinite(batch['data']).all()
        # Build the exact seven-stage, 141M-parameter architecture, without training it.
        network = cls.build_network_architecture(trainer.plans_manager, trainer.configuration_manager,
                                                3, 2, True)
        assert type(network).__name__ == 'ResidualEncoderUNet'
        assert sum(p.numel() for p in network.parameters()) == 141398092
        # Full network checkpoint exercises inherited nnUNet save/load and epoch state.
        trainer.network = network
        trainer.optimizer = torch.optim.SGD(network.parameters(), lr=.01, momentum=.99)
        if name == 'nnUNetTrainerMAMAGuardA5V1':
            # One bounded CPU update through the full shared architecture.
            prediction = network(torch.randn(1, 3, 32, 128, 128))
            outputs = prediction if isinstance(prediction, (list, tuple)) else [prediction]
            loss = sum(p.square().mean() for p in outputs)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            del prediction, outputs, loss
        trainer.was_initialized = True
        trainer.current_epoch = 6
        trainer.inference_allowed_mirroring_axes = (0,1,2)
        checkpoint = tmp_path/(name+'.pth')
        trainer.save_checkpoint(str(checkpoint))
        first = next(network.parameters())
        expected = first.detach().clone()
        with torch.no_grad():
            first.add_(1)
        trainer.load_checkpoint(str(checkpoint))
        torch.testing.assert_close(first, expected)
        assert trainer.current_epoch == 7
        if name == 'nnUNetTrainerMAMAGuardA5V1':
            assert trainer.optimizer.state
        assert name in trainer.output_folder
        del trainer, network
