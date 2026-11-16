import numpy as np
import pytest

torch = pytest.importorskip("torch")
from mamaguardnet.augmentation.tensor_affine import apply_affine  # noqa: E402


def test_shared_affine_accepts_numpy_fields_and_preserves_tensor_contract():
    image = torch.stack([torch.zeros(5, 7, 9), torch.ones(5, 7, 9),
                         torch.full((5, 7, 9), 3.0)])
    gain = np.full((1, 5, 7, 9), 1.7)
    result = apply_affine(image, gain, -0.3)
    assert result.dtype == image.dtype
    assert result.device == image.device
    assert result.shape == image.shape
    assert torch.all(result[2] > result[1])
    assert torch.all(result[1] > result[0])
    torch.testing.assert_close(result[1] - result[0], torch.full_like(image[0], 1.7))
