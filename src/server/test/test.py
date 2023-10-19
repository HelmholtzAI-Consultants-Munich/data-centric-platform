import numpy as np
import torch

from tqdm import tqdm

import sys
import cv2 
import os

from copy import deepcopy

import sys
sys.path.append("../")

from models import CellposePatchCNN
from dcp_server.utils import read_config
from skimage.color import label2rgb

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



if __name__=='__main__':

    img = cv2.imread("/home/ubuntu/data-centric-platform/src/server/dcp_server/data/img.jpg", cv2.IMREAD_GRAYSCALE)
    msk = np.load("/home/ubuntu/data-centric-platform/src/server/dcp_server/data/mask.npy")
    classifier_model_config, classifier_train_config, classifier_eval_config = {}, {}, {}

    segmentor_model_config = read_config("model", config_path = "config.cfg")
    segmentor_train_config = read_config("train", config_path = "config.cfg")
    segmentor_eval_config = read_config("eval", config_path = "config.cfg")

    patch_model = CellposePatchCNN(
        segmentor_model_config, segmentor_train_config, segmentor_eval_config,
        classifier_model_config, classifier_train_config, classifier_eval_config)
    
    images, masks = get_dataset("/home/ubuntu/data-centric-platform/src/server/dcp_server/data")

    # for i in tqdm(range(1)):
    #     loss_train = patch_model.train(deepcopy(images), deepcopy(masks))
    # assert(loss_train>1e-2)

    # instance segmentation mask (C, W, H) --> semantic multiclass segmentation mask (W, H)
    # for i in range(msk.shape[0]):
    #     msk[i, ...][msk[i, ...] > 0] = i + 1

    # msk = msk.sum(0)

    # img = img.mean(axis=1, keepdims=True)

    final_mask, jaccard_index = patch_model.eval(img, instance_mask=msk)
    final_mask = final_mask.numpy()

    cv2.imwrite("/home/ubuntu/data-centric-platform/src/server/dcp_server/data/final_mask.jpg", 255*label2rgb(final_mask))
    print(jaccard_index)



    

    