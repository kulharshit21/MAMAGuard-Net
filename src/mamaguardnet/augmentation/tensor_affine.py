"""Tensor-compatible affine operation for batchgenerators v2 image transforms."""
import torch


def apply_affine(image: torch.Tensor, gain, offset) -> torch.Tensor:
    gain = torch.as_tensor(gain, device=image.device, dtype=image.dtype)
    offset = torch.as_tensor(offset, device=image.device, dtype=image.dtype)
    return image * gain + offset
