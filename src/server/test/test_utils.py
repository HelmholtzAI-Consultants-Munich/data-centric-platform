import numpy as np
import pytest
from dcp_server.utils.processing import find_max_patch_size


@pytest.fixture
def sample_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 3:7] = 1
    mask[7:9, 2:5] = 1
    return mask


def test_find_max_patch_size(sample_mask):
    # Test when the function is called with a sample mask
    result = find_max_patch_size(sample_mask)
    assert isinstance(result, float)
    assert result > 0
