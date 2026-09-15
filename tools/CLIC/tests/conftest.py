"""What every CLIC test needs: the package on the path, a scan writer, and a
stand-in for the network so nothing here wants a GPU or the 176 MB checkpoint.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

import sadt_clic  # noqa: E402


@pytest.fixture
def make_scan():
    """Write a NIfTI volume and return its path.

    `affine` is a real argument because the geometry tests need a scan that is
    NOT axis-aligned at unit spacing: an identity affine hides every mistake a
    copied header could make.
    """
    def _make(path, shape=(8, 8, 3), affine=None, data=None, dtype=np.float32):
        import nibabel as nib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if data is None:
            data = np.random.default_rng(0).random(shape).astype(dtype)
        if affine is None:
            affine = np.eye(4)
        nib.save(nib.Nifti1Image(data, affine), str(path))
        return path

    return _make


@pytest.fixture
def stub_network(monkeypatch):
    """Replace the network only, so everything around it runs for real.

    `resolve_device` is deliberately left alone: the tests that care which
    device was chosen need the real one, with `torch.cuda.is_available` moved
    under it.
    """
    def build_model(checkpoint_path, device):
        return object(), 4

    def segment_volume(model, volume, device, score_threshold):
        labels = np.zeros(volume.shape, dtype=np.int16)
        labels[0, 0, 0] = 3
        return labels, 1

    monkeypatch.setattr(sadt_clic, "build_model", build_model)
    monkeypatch.setattr(sadt_clic, "segment_volume", segment_volume)


@pytest.fixture
def stubbed(stub_network, monkeypatch):
    """`stub_network`, plus a device that never looks for a card."""
    monkeypatch.setattr(sadt_clic, "resolve_device", lambda requested: "cpu")


@pytest.fixture
def no_cuda(monkeypatch):
    """No CUDA device is visible, whatever this machine actually has.

    The development machine has a card, so a fallback test that did not do this
    would pass by accident there and fail in CI.
    """
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def with_cuda(monkeypatch):
    """A CUDA device is visible, whatever this machine actually has."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
