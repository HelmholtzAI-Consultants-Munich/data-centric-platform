#models


from cellpose import models, utils

import torch
from torch import nn

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

#from segment_anything import SamPredictor, sam_model_registry
#from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator


class CustomCellposeModel(models.CellposeModel):
    """Custom cellpose model inheriting the attributes and functions from the original CellposeModel and implementing
    additional attributes and methods needed for this project.
    """    
    def __init__(self, model_config, train_config, eval_config):
        """Constructs all the necessary attributes for the CustomCellposeModel. 
        The model inherits all attributes from the parent class, the init allows to pass any other argument that the parent class accepts.
        Please, visit here https://cellpose.readthedocs.io/en/latest/api.html#id4 for more details on arguments accepted. 

        :param model_config: dictionary passed from the config file with all the arguments for the __init__ function and model initialization
        :type model_config: dict
        :param train_config: dictionary passed from the config file with all the arguments for training function
        :type train_config: dict
        :param eval_config: dictionary passed from the config file with all the arguments for eval function
        :type eval_config: dict
        """
        
        # Initialize the cellpose model
        super().__init__(**model_config)
        self.train_config = train_config
        self.eval_config = eval_config
        
    def eval(self, img, **eval_config):
        """Evaluate the model - find mask of the given image
        Calls the original eval function. 

        :param img: image to evaluate on
        :type img: np.ndarray
        :param z_axis: z dimension (optional, default is None)
        :type z_axis: int
        :return: mask of the image, list of 2D arrays, or single 3D array (if do_3D=True) labelled image.
        :rtype: np.ndarray
        """   
        return super().eval(x=img, **eval_config)[0] # 0 to take only mask
    
    def train(self, imgs, masks):
        """Trains the given model
        Calls the original train function.

        :param imgs: images to train on (training data)
        :type imgs: List[np.ndarray]
        :param masks: masks of the given images (training labels)
        :type masks: List[np.ndarray]
        """  
        super().train(train_data=imgs, train_labels=masks, **self.train_config)
    
    def masks_to_outlines(self, mask):
        """ get outlines of masks as a 0-1 array
        Calls the original cellpose.utils.masks_to_outlines function

        :param mask: int, 2D or 3D array, mask of an image
        :type mask: ndarray
        :return: outlines
        :rtype: ndarray
        """        
        return utils.masks_to_outlines(mask) #[True, False] outputs

class CellFullyConvClassifier(nn.Module):
    
    '''
    Fully convolutional classifier for cell images.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
    '''

    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.train_config = train_config
        self.eval_config = eval_config

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 2, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
        )

        self.final_conv = nn.Conv2d(128, num_classes, 1)

        self.pooling = nn.AdaptiveMaxPool2d(1)

     def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.final_conv(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        
        return x

    def train (self, imgs, labels):
    ## TODO should call forward repeatedly and perform the entire train loop
        
        """
            input:
            1) imgs - List[np.ndarray[np.uint8]] with shape (3, dx, dy)
            2) y - List[int]
        """

        lr = self.train_config.get('lr', 0.001)
        epochs = self.train_config.get('epochs', 1)
        batch_size = self.train_config.get('batch_size', 1)
        optimizer_class = self.train_config.get('optimizer', 'Adam')

        # Convert input images and labels to tensors
        imgs = [ torch.from_numpy(img) for img in imgs]
        labels = torch.tensor(labels)

        # Create a training dataset and dataloader
        train_dataset = TensorDataset(imgs, labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        # eval method evaluates a python string and returns an object, e.g. eval('print(1)') = 1
        # or eval('[1, 2, 3]') = [1, 2, 3]
        loss_fn = nn.CrossEntropyLoss()
        optimizer = eval(f'{optimizer_class}(lr={lr})')

        for _ in epochs:
            for i, data in enumerate(train_dataloader):

                imgs, labels = data
                optimizer.zero_grad()
                preds = self.forward(imgs)

                y_hats = torch.argmax(preds, 1)
                loss = loss_fn(y_hats, labels)
                loss.backward()

                optimizer.step()
    
    def eval(self, imgs):
    ## TODO should call forward once, model is in eval mode, and return predicted masks
        """
        Evaluate the model on the provided images and return predicted labels.
            Input:
            imgs: List[np.ndarray[np.uint8]] with shape (3, dx, dy)
            Output:
            labels: List of predicted labels.
        """
        labels = []
        for img in imgs:
            
            img = torch.from_numpy(img).unsqueeze(0)
            labels.append(self.forward(img))

        return labels
# class CustomSAMModel():
# # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
#     def __init__(self):
#         pass


class CellposePatchCNN():

    """
    Cellpose & patches of cells and then cnn to classify each patch
    """
    
    def __init__(self,  model_config, train_config, eval_config ):

        
        self.train_config = train_config
        self.eval_config = eval_config

        # Initialize the classifier and the cellpose model 
        self.classifier = CellFullyConvClassifier()
        self.segmentor = CustomCellposeModel(model_config, train_config, eval_config)

    def init_from_checkpoints(self, chpt_classifier=None, chpt_segmentor=None):
    """
    Initialize the model from pre-trained checkpoints.
    """

        self.segmentor = CustomCellposeModel(
            model_config={"gpu":torch.cuda.is_available(), "pretrained_model":chpt_segmentor} 
            )
        self.classifier.load_state_dict(torch.load(chpt_classifier)["model"])


    def train(self, imgs, masks, **train_config):

        # masks should have first channel as a cellpose mask and
        # all other layers corresponds to the classes
        ## TODO: take care of the images and masks preparation -> this step isn't for now @Mariia
        ## TODO call create_patches (adjust create_patch_dataset function)
        ## to prepare imgs and masks for training CNN
        ## TODO call self.classifier.train(imgs, masks)

        black_bg = train_config.get("black_bg", False)
        include_mask = train_config.get("include_mask", False)
        include_cellpose_mask = train_config.get("include_cellpose_mask", True)

        self.segmentor.train(imgs, masks)
        patches, labels = self.create_patch_dataset(self, imgs, masks, black_bg:bool, include_mask:bool, **kwargs)
        self.classifier.train(patches, labels)


    def eval(self, img, **eval_config):
        
        ## TODO implement the eval pipeline, i.e. first call self.segmentor.eval, then split again into patches
        ## using resulting seg and then call self.classifier.eval. The final mask which is returned should have 
        ## first channel the output of cellpose and the rest are the class channels
        mask = self.segmentor.eval(img)
        result = self.get_prediction(img, mask)

        return result

    def get_prediction(self, input_image, cellpose_mask, model_classifier):
    """
        Performs object segmentation and classification on an input image using the Cellpose model and a classifier model.

        Args:
            image_path (str): The file path of the input image.
            model (CellposeModel): The Cellpose model used for object segmentation.
            model_classifier: The classifier model used for object classification.

        Returns:
            tuple: A tuple containing the cellpose_mask and final_mask, representing the segmentation masks obtained from
                the Cellpose model and the combined segmentation and classification mask, respectively.
    """

        # Obtain segmentation mask using Cellpose model

        # Find objects in the cellpose_mask
        locs = find_objects(cellpose_mask)

        # Get patches and labels based on object centroids
        patches, labels = get_centered_patches(input_image, cellpose_mask, int(1.5 * input_image.shape[0] // 5), noise_intensity=5)
        
        labels = torch.tensor(labels)
        labels_fit = []
        
        final_mask = torch.zeros(cellpose_mask.shape)

        with torch.no_grad():
            for i, patch in enumerate(patches):
                loc = locs[i]

                # Prepare image patch for classification
                img = torch.tensor(patch.astype(np.float32)).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0) / 255
                
                # Perform inference using model_classifier
                logits = self.classifier(img) 
        
                _, predicted = torch.max(logits, 1)
                labels_fit.append(predicted)

                # Assign predicted class to corresponding location in final_mask
                final_mask[loc] = predicted + 1

        # Apply mask to final_mask, retaining only regions where cellpose_mask is greater than 0
        final_mask = final_mask * ((cellpose_mask > 0).astype(np.uint8))
        
        return final_mask

    def find_max_patch_size(self, mask):

        # Find objects in the mask
        objects = ndi.find_objects(mask)

        # Initialize variables to store the maximum patch size
        max_patch_size = 0
        max_patch_indices = None

        # Iterate over the found objects
        for obj in objects:
            # Extract start and stop values from the slice object
            slices = [s for s in obj]
            start = [s.start for s in slices]
            stop = [s.stop for s in slices]

            # Calculate the size of the patch along each axis
            patch_size = tuple(stop[i] - start[i] for i in range(len(start)))

            # Calculate the total size (area) of the patch
            total_size = 1
            for size in patch_size:
                total_size *= size

            # Check if the current patch size is larger than the maximum
            if total_size > max_patch_size:
                max_patch_size = total_size
                max_patch_indices = obj
            
            max_patch_size_edge = np.ceil(np.sqrt(max_patch_size))

            return max_patch_size_edge

    def pad_centered_padded_patch(self, x: np.ndarray, c, p, mask: np.ndarray=None, noise_intensity=None) -> np.ndarray:
        """
        Crop a patch from an array `x` centered at coordinates `c` with size `p`, and apply padding if necessary.

        Args:
            x (np.ndarray): The input array from which the patch will be cropped.
            c (tuple): The coordinates (row, column, channel) at the center of the patch.
            p (tuple): The size of the patch to be cropped (height, width).

        Returns:
            np.ndarray: The cropped patch with applied padding.
        """

        height, width = p  # Size of the patch

        # Calculate the boundaries of the patch
        top = c[0] - height // 2
        bottom = top + height
        
        left = c[1] - width // 2
        right = left + width

        # Crop the patch from the input array

        if mask is not None:

            mask_ = mask.max(-1) if len(mask.shape) >= 3 else mask
            central_label = mask_[c[0], c[1]]
            # Zero out values in the patch where the mask is not equal to the central label
            m = (mask_ != central_label) & (mask_ > 0)
            x[m] = 0

            if noise_intensity is not None:
                x[m] = np.random.normal(scale=noise_intensity, size=x[m].shape)

        patch = x[max(top, 0):min(bottom, x.shape[0]), max(left, 0):min(right, x.shape[1])]
        
        if len(c) == 3:
            patch = patch[...,c[2]]
        
        # Calculate the required padding amounts

        size_x, size_y = x.shape[1], x.shape[0]

        # Apply padding if necessary
        if left < 0:
            patch = np.hstack((
                np.random.normal(scale=noise_intensity, size=(patch.shape[0], abs(left))).astype(np.uint8), patch
            ))
        
        # Apply padding on the right side if necessary
        if right > size_x:
            patch = np.hstack((
                patch, np.random.normal(scale=noise_intensity, size=(patch.shape[0], right - size_x)).astype(np.uint8)
            ))

        # Apply padding on the top side if necessary
        if top < 0:
            patch = np.vstack((
                np.random.normal(scale=noise_intensity, size=(abs(top), patch.shape[1])).astype(np.uint8), patch
            ))
        
        # Apply padding on the bottom side if necessary
        if bottom > size_y:
            patch = np.vstack((
                patch, np.random.normal(scale=noise_intensity, size=(bottom - size_y, patch.shape[1])).astype(np.uint8)
            ))

        return patch 


    def get_center_of_mass(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute the centers of mass for each object in a mask.

        Args:
            mask (np.ndarray): The input mask containing labeled objects.

        Returns:
            np.ndarray: An array of coordinates (row, column, channel) representing the centers of mass for each object.
        """
        # Compute the centers of mass for each labeled object in the mask
        centers_of_mass = [
        (int(x[0]), int(x[1]), int(x[2])) if mask.ndim >= 3 else (int(x[0]), int(x[1]), -1)
        for x in ndi.center_of_mass(mask, mask, np.arange(1, mask.max() + 1))
        ]

        return centers_of_mass


    def get_centered_patches(self, img, mask, p_size: int, noise_intensity=5):

        ''' 
        Extracts centered patches from the input image based on the centers of objects identified in the mask.

        Args:
            img: The input image.
            mask: The mask representing the objects in the image.
            p_size (int): The size of the patches to extract.
            noise_intensity: The intensity of noise to add to the patches.

        '''

        patches, labels = [], []

        centers_of_mass = get_center_of_mass(mask)
            # Crop patches around each center of mass and save them
        for i, c in enumerate(centers_of_mass):
            c_x, c_y, label = c
            
            patch = pad_centered_padded_patch(img.copy(), (c_x, c_y), (p_size, p_size), mask=mask, noise_intensity=noise_intensity)

            patches.append(patch)
            labels.append(label)

        return patches, labels

    def create_patch_dataset(self, imgs, masks, black_bg:bool, include_mask:bool, **kwargs):
        '''
        TODO: Split img and masks into patches of equal size which are centered around the cells.
        The algorithm should first run through all images to find the max cell size, and use
        the max cell size to define the patch size. All patches and masks should then be returned
        in the same format as imgs and masks (same type, i.e. check if tensor or np.array and same 
        convention of dims, e.g.  CxHxW)
        Args:
            imgs ():
            masks ():
            black_bg (bool): Flag indicating whether to use a black background for patches.
            include_mask (bool): Flag indicating whether to include the mask along with patches.
        '''

        noise_intensity = kwargs.get("data", {}).get("noise_intensity", 5)

        #max_patch_size = np.max([self.find_max_patch_size(mask) in masks])
        max_patch_size = kwargs.get("data", {}).get("patch_size", 64)
        num_of_channels = kwargs.get("data", {}).get("num_classes", 3)

        patches, labels = [], []
        is_train = masks[0].ndim == 3

        for img, msk in zip(imgs, masks):
            
            # training dataset
            if is_train:
                # mask has dimension WxHxNum_of_channels
                for channel in range(num_of_channels):

                    patch, label = get_centered_patches(img, msk, int(1.5 * img.shape[0] // 5), 
                                                            noise_intensity=noise_intensity)
                    
                    patches.append(patch)
                    labels.append(label)

            # test dataset
            # mask has dimention WxH 
            else:
                patch, _ = get_centered_patches(img, msk, int(1.5 * img.shape[0] // 5), 
                                                            noise_intensity=noise_intensity)
                patches.append(patch)

        if is_train:
            return patches, labels

        else:
            return patches