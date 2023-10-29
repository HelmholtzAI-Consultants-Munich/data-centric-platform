import sys
import torch 
from torchmetrics import JaccardIndex
import numpy as np

sys.path.append(".")

from dcp_server.models import CellposePatchCNN
from dcp_server.utils import read_config
from synthetic_dataset import get_synthetic_dataset

import pytest

@pytest.fixture
def patch_model():
    

    model_config = read_config('model', config_path='dcp_server/config.cfg')
    train_config = read_config('train', config_path='dcp_server/config.cfg')
    eval_config = read_config('eval', config_path='dcp_server/config.cfg')

    patch_model = CellposePatchCNN(model_config, train_config, eval_config)
    return patch_model

@pytest.fixture
def data_train():
    images, masks = get_synthetic_dataset(num_samples=2)
    return images, masks

@pytest.fixture
def data_eval(): 
    img, msk = get_synthetic_dataset(num_samples=1)
    return img, msk

def test_train_run(data_train, patch_model):

    images, masks = data_train

    patch_model.train(images, masks)
    # CellposeModel eval doesn't work when images is a list (only on a single image)
    # assert(patch_model.segmentor.loss>1e-2) #TODO figure out appropriate value
    assert(patch_model.classifier.loss>1e-2)
    
def test_eval_run(data_eval, patch_model):

    imgs, masks = data_eval
    jaccard_index_instances = 0
    jaccard_index_classes = 0

    jaccard_metric_binary = JaccardIndex(task="multiclass", num_classes=2, average="macro", ignore_index=0)
    jaccard_metric_multi = JaccardIndex(task="multiclass", num_classes=4, average="macro", ignore_index=0)

    for img, mask in zip(imgs, masks):

        #mask - instance multiclass segmentation (512, 512, 3)
        #pred_mask - tuple of cellpose (512, 512), patch net multiclass segmentation (512, 512, 3)

        pred_mask = patch_model.eval(img) #, channels=[0,0])
        mask = (mask > 0) * np.arange(1, 4)

        pred_mask_bin = torch.tensor(pred_mask[0].astype(bool).astype(int))
        bin_mask = torch.tensor(mask.sum(-1).astype(bool).astype(int))

        jaccard_index_instances += jaccard_metric_binary(
            pred_mask_bin, 
            bin_mask
        )
        jaccard_index_classes += jaccard_metric_multi(
            torch.tensor(pred_mask[1].astype(int)), 
            torch.tensor(mask.sum(-1).astype(int))
        )
    
    jaccard_index_instances /= len(imgs)
    assert(jaccard_index_instances<0.6)
    jaccard_index_classes /= len(imgs)
    assert(jaccard_index_instances<0.6)
    







    

    