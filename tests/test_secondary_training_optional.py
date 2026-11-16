"""Executable CPU integration checks; skips explicitly without training dependencies."""
import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec('nnunetv2') is None,
                                reason='Isolated nnUNet training environment unavailable')


def test_transforms_forward_backward_and_checkpoint(tmp_path):
    import torch

    from mamaguardnet.nnunet.secondary_trainers_v1 import AdaptiveV1, IndependentSmoothV1
    torch.set_num_threads(1)
    x = torch.randn(3, 8, 12, 12)
    for transform in [IndependentSmoothV1(), AdaptiveV1(), AdaptiveV1('shifted')]:
        transformed = transform(image=x)['image']
        assert transformed.shape == x.shape and torch.isfinite(transformed).all()
        model = torch.nn.Conv3d(3, 2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        loss = model(transformed[None]).square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        checkpoint = tmp_path/'cpu_fixture.pth'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint)
        restored = torch.nn.Conv3d(3, 2, 1)
        state = torch.load(checkpoint, weights_only=True)
        restored.load_state_dict(state['model'])
        torch.testing.assert_close(restored(transformed[None]), model(transformed[None]))
    # This fixture does not certify full nnUNet trainer resume or architecture discovery.
