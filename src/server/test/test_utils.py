import numpy as np
import pytest
from dcp_server.utils.processing import get_objects, normalise, pad_image


@pytest.fixture
def sample_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 3:7] = 1
    mask[7:9, 2:5] = 1
    return mask


def test_get_objects(sample_mask):
    """Test finding labeled connected components / bounding boxes in a mask."""
    result = get_objects(sample_mask)
    assert isinstance(result, list)
    assert len(result) >= 1


def test_normalise_minmax():
    """Test min-max normalisation."""
    img = np.array([[0.0, 50.0], [100.0, 200.0]])
    result = normalise(img, norm="min-max")
    assert result.min() == 0.0
    assert result.max() == 1.0
    assert result.shape == img.shape


def test_pad_image():
    """Test image padding for divisibility."""
    img = np.zeros((10, 15))
    result = pad_image(img, height=10, width=15, dividable=16)
    assert result.shape[0] >= 10
    assert result.shape[1] >= 15
    assert result.shape[0] % 16 == 0
    assert result.shape[1] % 16 == 0
