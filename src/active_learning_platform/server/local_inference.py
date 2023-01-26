import os
from pathlib import Path
import torch 
from cellpose import models
from skimage.io import imread, imsave
from skimage.transform import resize, rescale
import sys


def run_inference(eval_data_path, accepted_types):
    ''' 
    Runs inference for the images in uncurated data directory. 
    Currently, cellpose is used for segmentation.
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device=="cuda":
        model = models.Cellpose(gpu=True, model_type="cyto")
    else:
        model = models.Cellpose(gpu=False, model_type="cyto")
    
    list_files = [file for file in os.listdir(eval_data_path) if Path(file).suffix in accepted_types]
    
    list_of_files_not_supported = []

    for img_filename in list_files:
        # don't do this for segmentations in the folder
        if '_seg' in img_filename:  
            continue
            #extend to check the prefix also matches an existing image
            #seg_name = Path(self.img_filename).stem+'_seg'+Path(self.img_filename).suffix
        else:
            img = imread(os.path.join(eval_data_path, img_filename)) #DAPI
            orig_size = img.shape
            seg_name = Path(img_filename).stem+'_seg.tiff' #+Path(img_filename).suffix

            # png and jpeg will be RGB by deafult and 2D
            # tif can be grayscale 2D
            # or 2D RGB and RGBA
            if Path(img_filename).suffix in (".jpg", ".jpeg", ".png") or (Path(img_filename).suffix in (".tiff", ".tif") and len(orig_size)==2 or (len(orig_size)==3 and (orig_size[-1]==3 or orig_size[-1]==4))):
                height, width = orig_size[0], orig_size[1]
                max_dim  = max(height, width)
                rescale_factor = max_dim/512
                img = rescale(img, 1/rescale_factor)
                mask, _, _, _ = model.eval(img)
                mask = resize(mask, (height, width), order=0)

            # or 3D tiff grayscale 
            elif Path(img_filename).suffix in (".tiff", ".tif") and len(orig_size)==3:
                print('Warning: 3D image stack found. We are assuming your first dimension is your stack dimension. Please cross check this.')
                height, width = orig_size[1], orig_size[2]
                max_dim = max(height, width)
                rescale_factor = max_dim/512
                img = rescale(img, 1/rescale_factor, channel_axis=0)
                mask, _, _, _ = model.eval(img, z_axis=0)
                mask = resize(mask, (orig_size[0], height, width), order=0)
                
            else: 
                list_of_files_not_supported.append(img_filename)

            imsave(os.path.join(eval_data_path, seg_name), mask)
            
    return list_of_files_not_supported