import numpy as np
import pytest
from dcp_client.utils.compute4mask import Compute4Mask


@pytest.fixture
def sample_data():
    instance_mask = np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [2, 2, 0, 0, 0],
            [0, 0, 3, 3, 0],
        ]
    )
    labels_mask = np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [2, 2, 0, 0, 0],
            [0, 0, 1, 1, 0],
        ]
    )
    return instance_mask, labels_mask


def test_get_unique_objects(sample_data):
    instance_mask, _ = sample_data
    unique_objects = Compute4Mask.get_unique_objects(instance_mask)
    assert unique_objects == [1, 2, 3]


def test_get_contours(sample_data):
    instance_mask, _ = sample_data
    contour_mask = Compute4Mask.get_contours(instance_mask)
    assert contour_mask.shape == instance_mask.shape
    assert contour_mask[0, 1] == 1  # randomly check a contour location is present


def test_add_contour(sample_data):
    instance_mask, labels_mask = sample_data
    contours_mask = Compute4Mask.get_contours(instance_mask, contours_level=0.1)
    labels_mask_wo_contour = np.copy(labels_mask)
    labels_mask_wo_contour[contours_mask != 0] = 0
    updated_labels_mask = Compute4Mask.add_contour(
        labels_mask_wo_contour, instance_mask
    )
    assert np.array_equal(updated_labels_mask[:3], labels_mask[:3])


def test_compute_new_instance_mask(sample_data):
    instance_mask, labels_mask = sample_data
    labels_mask[labels_mask == 1] = 0
    updated_instance_mask = Compute4Mask.compute_new_instance_mask(
        labels_mask, instance_mask
    )
    assert list(np.unique(updated_instance_mask)) == [0, 2]


def test_compute_new_labels_mask_obj_added(sample_data):
    instance_mask, labels_mask = sample_data
    original_instance_mask = np.copy(instance_mask)
    instance_mask[0, 0] = 4
    old_instances = Compute4Mask.get_unique_objects(original_instance_mask)
    new_labels_mask = Compute4Mask.compute_new_labels_mask(
        labels_mask, instance_mask, original_instance_mask, old_instances
    )
    assert new_labels_mask[0, 0] == 4


def test_compute_new_labels_mask_obj_erased(sample_data):
    instance_mask, labels_mask = sample_data
    original_instance_mask = np.copy(instance_mask)
    instance_mask[0] = 0
    old_instances = Compute4Mask.get_unique_objects(original_instance_mask)
    new_labels_mask = Compute4Mask.compute_new_labels_mask(
        labels_mask, instance_mask, original_instance_mask, old_instances
    )
    assert np.all(new_labels_mask[0]) == 0
    assert np.array_equal(new_labels_mask[1:], labels_mask[1:])


def test_compute_new_labels_mask_obj_added(sample_data):
    instance_mask, labels_mask = sample_data
    original_instance_mask = np.copy(instance_mask)
    instance_mask[:, -1] = 1
    old_instances = Compute4Mask.get_unique_objects(original_instance_mask)
    new_labels_mask = Compute4Mask.compute_new_labels_mask(
        labels_mask, instance_mask, original_instance_mask, old_instances
    )
    assert np.all(new_labels_mask[:, -1]) == 1


def assert_consistent_labels(sample_data):
    instance_mask, labels_mask = sample_data
    user_annot_error, mask_mismatch_error, faulty_ids_annot, faulty_ids_missmatch = (
        Compute4Mask.assert_consistent_labels(sample_data)
    )
    assert user_annot_error == False
    assert mask_mismatch_error == False
    assert len(faulty_ids_annot) == len(faulty_ids_missmatch) == 0
    instance_mask[instance_mask == 3] = 1
    labels_mask[1, 2] = 2
    user_annot_error, mask_mismatch_error, faulty_ids_annot, faulty_ids_missmatch = (
        Compute4Mask.assert_consistent_labels(np.stack(instance_mask, labels_mask))
    )
    assert user_annot_error == True
    assert mask_mismatch_error == True
    assert len(faulty_ids_annot) == 1
    assert faulty_ids_annot[0] == 1
    assert len(faulty_ids_missmatch) == 1
    assert faulty_ids_missmatch[0] == 1
