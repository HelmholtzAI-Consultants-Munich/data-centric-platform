import sys

sys.path.append(".")

from glob import glob
import pytest

# import inspect

import random
import numpy as np
import torch
from torchmetrics import JaccardIndex

from dcp_server.models import *
from dcp_server.utils.helpers import read_config
from synthetic_dataset import get_synthetic_dataset

seed_value = 2023
random.seed(seed_value)
torch.manual_seed(seed_value)
np.random.seed(seed_value)

model_mapping = {
    "CustomCellpose": CustomCellpose,
    "Inst2MultiSeg": Inst2MultiSeg,
    "MultiCellpose": MultiCellpose,
    "UNet": UNet,
}

config_paths = glob("test/configs/*.yaml")


@pytest.fixture(params=config_paths)
def config_path(request):
    return request.param


@pytest.fixture()
# def model(model_class, config_path):
def model(config_path):

    setup_config = read_config("setup", config_path=config_path)
    model_config = read_config("model", config_path=config_path)
    data_config = read_config("data", config_path=config_path)
    eval_config = read_config("eval", config_path=config_path)

    model_name = setup_config["model_to_use"]
    model_class = model_mapping.get(model_name)
    model = model_class(
        model_name, model_config, data_config, eval_config
    )
    # str(model_class)
    return model


@pytest.fixture
def data_train():
    images, masks = get_synthetic_dataset(num_samples=4, canvas_size=(512, 768))
    masks = [np.array(mask) for mask in masks]
    masks_instances = [mask.sum(-1) for mask in masks]
    masks_classes = [((mask > 0) * np.arange(1, 4)).sum(-1) for mask in masks]
    masks_ = [
        np.stack((instances, classes))
        for instances, classes in zip(masks_instances, masks_classes)
    ]
    return images, masks_


@pytest.fixture
def data_eval():
    img, msk = get_synthetic_dataset(num_samples=1)
    msk = np.array(msk)
    msk_ = np.stack(
        (msk.sum(-1), ((msk > 0) * np.arange(1, 4)).sum(-1)), axis=0
    ).transpose(1, 0, 2, 3)
    return img, msk_


# def test_train_eval_run(data_train, data_eval, model):
#     """
#     Performs testing, training, and evaluation with the provided data and model.
#     """
#     # train
#     images, masks = data_train
#     if model.model_name == "CustomCellpose":
#         masks = [mask[0] for mask in masks]
#     model.train(images, masks)

#     # retrieve the attribute names of the class of the current model
#     attrs = model.__dict__.keys()

#     if "metric" in attrs:
#         assert model.metric > 0.1
#     if "loss" in attrs:
#         assert model.loss < 0.83

#     # validate
#     imgs_test, masks_test = data_eval
#     if model.model_name == "CustomCellpose":
#         masks = [mask[0] for mask in masks_test]

#     jaccard_index_instances = 0
#     jaccard_index_classes = 0

#     jaccard_metric_binary = JaccardIndex(
#         task="multiclass", num_classes=2, average="macro", ignore_index=0
#     )
#     jaccard_metric_multi = JaccardIndex(
#         task="multiclass", num_classes=4, average="macro", ignore_index=0
#     )

#     for img, mask in zip(imgs_test, masks_test):

#         # mask - instance segmentation mask + classes (2, 512, 512)
#         # pred_mask - tuple of cellpose (512, 512), patch net multiclass segmentation (512, 512, 2)

#         pred_mask = model.eval(img)

#         if pred_mask.ndim > 2:
#             pred_mask_bin = torch.tensor((pred_mask[0] > 0).astype(bool).astype(int))
#         else:
#             pred_mask_bin = torch.tensor((pred_mask > 0).astype(bool).astype(int))

#         bin_mask = torch.tensor((mask[0] > 0).astype(bool).astype(int))

#         jaccard_index_instances += jaccard_metric_binary(pred_mask_bin, bin_mask)

#         if pred_mask.ndim > 2:

#             jaccard_index_classes += jaccard_metric_multi(
#                 torch.tensor(pred_mask[1].astype(int)),
#                 torch.tensor(mask[1].astype(int)),
#             )

#     jaccard_index_instances /= len(imgs_test)
#     assert jaccard_index_instances > 0.2

#     if pred_mask.ndim > 2:

#         jaccard_index_classes /= len(imgs_test)
#         assert jaccard_index_classes > 0.1


# def test_train_run(data_train, model):

#     images, masks = data_train
#     model.train(images, masks)
#     # assert(patch_model.segmentor.loss>1e-2) # TODO figure out appropriate value

#     # retrieve the attribute names of the class of the current model
#     attrs = model.__dict__.keys()

#     if "classifier" in attrs:
#         assert(model.classifier.loss<0.4)
#     if "metric" in attrs:
#         assert(model.metric>0.1)
#     if "loss" in attrs:
#         assert(model.loss<0.3)

# def test_eval_run(data_train, data_eval, model):

#     images, masks = data_train
#     model.train(images, masks)

#     imgs_test, masks_test = data_eval

#     jaccard_index_instances = 0
#     jaccard_index_classes = 0

#     jaccard_metric_binary = JaccardIndex(task="multiclass", num_classes=2, average="macro", ignore_index=0)
#     jaccard_metric_multi = JaccardIndex(task="multiclass", num_classes=4, average="macro", ignore_index=0)

#     for img, mask in zip(imgs_test, masks_test):

#         #mask - instance segmentation mask + classes (2, 512, 512)
#         #pred_mask - tuple of cellpose (512, 512), patch net multiclass segmentation (512, 512, 2)

#         pred_mask = model.eval(img) #, channels=[0,0])

#         if pred_mask.ndim > 2:
#             pred_mask_bin = torch.tensor((pred_mask[0]>0).astype(bool).astype(int))
#         else:
#             pred_mask_bin = torch.tensor((pred_mask > 0).astype(bool).astype(int))

#         bin_mask = torch.tensor((mask[0]>0).astype(bool).astype(int))

#         jaccard_index_instances += jaccard_metric_binary(
#             pred_mask_bin,
#             bin_mask
#         )

#         if pred_mask.ndim > 2:

#             jaccard_index_classes += jaccard_metric_multi(
#                 torch.tensor(pred_mask[1].astype(int)),
#                 torch.tensor(mask[1].astype(int))
#             )

#     jaccard_index_instances /= len(imgs_test)
#     assert(jaccard_index_instances>0.2)

#     # for PatchCNN model
#     if pred_mask.ndim > 2:

#         jaccard_index_classes /= len(imgs_test)
#         assert(jaccard_index_classes>0.1)
