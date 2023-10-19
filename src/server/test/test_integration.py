import os
import cv2 
import sys
import torch 
from torchmetrics import JaccardIndex


import numpy as np
from tqdm import tqdm
from copy import deepcopy
from skimage.color import label2rgb

sys.path.append("../")
from dcp_server.models import CellposePatchCNN
from dcp_server.utils import read_config

import pytest


def get_dataset(dataset_path):

    images_path = os.path.join(dataset_path, "images")
    masks_path = os.path.join(dataset_path, "masks")


    images_files = [img for img in os.listdir(images_path)]
    masks_files = [mask for mask in os.listdir(masks_path)]

    images, masks = [], []
    for img_file, mask_file in zip(images_files, masks_files):

        img_path = os.path.join(images_path, img_file)
        mask_path = os.path.join(masks_path, mask_file)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        msk = np.load(mask_path)

        images.append(img)
        masks.append(msk)

    return images, masks


@pytest.fixture
def patch_model():

    model_config = read_config('model', config_path='config.cfg')
    train_config = read_config('train', config_path='config.cfg')
    eval_config = read_config('eval', config_path='config.cfg')

    patch_model = CellposePatchCNN(model_config, train_config, eval_config)
    return patch_model

@pytest.fixture
def data_train():
    images, masks = get_dataset("/home/ubuntu/data-centric-platform/src/server/dcp_server/data")
    return images, masks

@pytest.fixture
def data_eval():
    img = cv2.imread("/home/ubuntu/data-centric-platform/src/server/dcp_server/data/img.jpg", cv2.IMREAD_GRAYSCALE)
    msk = np.load("/home/ubuntu/data-centric-platform/src/server/dcp_server/data/mask.npy")
    return img, msk

def test_train_run(data_train, patch_model):

    images, masks = data_train
    patch_model.train(images, masks)
    assert(patch_model.segmentor.loss>1e-2) #TODO figure out appropriate value
    assert(patch_model.classifier.loss>1e-2)
    
def test_eval_run(data_eval, patch_model):

    imgs, masks = data_eval
    jaccard_index_instances = 0
    jaccard_index_classes = 0
    for img, mask in zip(imgs, masks):
        pred_mask = patch_model.eval(img)
        jaccard_index_instances += JaccardIndex(pred_mask[0], mask[0])
        jaccard_index_classes += JaccardIndex(pred_mask[1], mask[1])
    
    jaccard_index_instances /= len(imgs)
    assert(jaccard_index_instances<0.6)
    jaccard_index_classes /= len(imgs)
    assert(jaccard_index_instances<0.6)
    







    

    