from cellpose import models, utils
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from scipy.ndimage import find_objects, center_of_mass

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
        super().train(train_data=deepcopy(imgs), train_labels=masks, **self.train_config)
        self.loss = self.loss_fn(masks, super().eval(imgs, self.eval_config)[0])
    
    def masks_to_outlines(self, mask):
        """ get outlines of masks as a 0-1 array
        Calls the original cellpose.utils.masks_to_outlines function

        :param mask: int, 2D or 3D array, mask of an image
        :type mask: ndarray
        :return: outlines
        :rtype: ndarray
        """        
        return utils.masks_to_outlines(mask) #[True, False] outputs

class CellClassifierFCNN(nn.Module):
    
    '''
    Fully convolutional classifier for cell images.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
    '''

    def __init__(self, model_config, train_config, eval_config):
        super().__init__()

        self.in_channels = model_config["in_channels"]
        self.num_classes = model_config["num_classes"] + 1

        self.train_config = train_config
        self.eval_config = eval_config
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, 3, 2, 5),
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
        self.final_conv = nn.Conv2d(128, self.num_classes, 1)
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
        """
            input:
            1) imgs - List[np.ndarray[np.uint8]] with shape (3, dx, dy)
            2) labels - List[int]
        """

        lr = self.train_config['lr']
        epochs = self.train_config['n_epochs']
        batch_size = self.train_config['batch_size']
        # optimizer_class = self.train_config['optimizer']

        # Convert input images and labels to tensors
        imgs = [(img-np.min(img))/(np.max(img)-np.min(img)) for img in imgs]
        # convert to tensor
        imgs = torch.stack([torch.from_numpy(img.astype(np.float32)) for img in imgs])
        imgs = torch.permute(imgs, (0, 3, 1, 2)) 
        # Your classification label mask
        labels = torch.LongTensor([label for label in labels])
        # Convert to one-hot encoding
        #labels = torch.nn.functional.one_hot(labels, self.num_classes)

        # Create a training dataset and dataloader
        train_dataset = TensorDataset(imgs, labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(params=self.parameters(), lr=lr) #eval(f'{optimizer_class}(params={self.parameters()}, lr={lr})')
        # TODO check if we should replace self.parameters with super.parameters()
        
        for _ in tqdm(range(epochs), desc="Running CellClassifierFCNN training"):
            self.loss = 0
            for data in train_dataloader:
                imgs, labels = data
                optimizer.zero_grad()
                preds = self.forward(imgs)
                l = loss_fn(preds, labels)
                l.backward()
                optimizer.step()
                self.loss += l.item()

            self.loss /= len(train_dataloader) 
    
    def eval(self, img):
        """
        Evaluate the model on the provided image and return the predicted label.
            Input:
            img: np.ndarray[np.uint8]
            Output: y_hat - The predicted label
        """ 
        # normalise
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        # convert to tensor
        img = torch.permute(torch.tensor(img.astype(np.float32)), (2, 0, 1)).unsqueeze(0) 
        preds = self.forward(img)
        y_hat = torch.argmax(preds, 1)
        return y_hat

class CellposePatchCNN():

    """
    Cellpose & patches of cells and then cnn to classify each patch
    """
    
    def __init__(self, model_config, train_config, eval_config):
        
        self.model_config = model_config
        self.train_config = train_config
        self.eval_config = eval_config

        # Initialize the cellpose model and the classifier
        self.segmentor = CustomCellposeModel(self.model_config["segmentor"], 
                                             self.train_config["segmentor"],
                                             self.eval_config["segmentor"])
        self.classifier = CellClassifierFCNN(self.model_config["classifier"],
                                             self.train_config["classifier"],
                                             self.eval_config["classifier"])

    def init_from_checkpoints(self, chpt_classifier=None, chpt_segmentor=None):
        """
        Initialize the model from pre-trained checkpoints.
        """

        self.segmentor = CustomCellposeModel(
            model_config={"gpu":torch.cuda.is_available(), "pretrained_model":chpt_segmentor} 
            )
        self.classifier.load_state_dict(torch.load(chpt_classifier)["model"])


    def train(self, imgs, masks):
        # masks should have first channel as a cellpose mask and all other layers
        # correspond to the classes, to prepare imgs and masks for training CNN

        # TODO I commented below lines. I think we should expect masks to have same size as output of eval, i.e. one channel instances, second channel classes
        # in this case the shape of masks shall be [2, H, W] or [2, 3, H, W] for 3D
        # In this case remove commented lines
        # num_classes = self.model_config["classifier"]["num_classes"]
        # masks_1channel = [mask.sum(0) for mask in masks]
        # masks_classifier = [mask if mask.shape[-1] == num_classes else
                                 # mask.transpose(1, 2, 0) for mask in masks]
        
        # train cellpose
        masks = np.array(masks)
        masks_instances = list(masks[:,0, ...])
        self.segmentor.train(imgs, masks_instances)
        # create patch dataset to train classifier
        masks_classes = list(masks[:,1, ...])
        patches, labels = self.create_patch_dataset(imgs, masks_classes, masks_instances)
        # train classifier
        self.classifier.train(patches, labels)
        #return # TODO - define if we need to return something

    def eval(self, img, **eval_config):

        # TBD we assume image is either 2D [H, W] or 3D [H, W, C] (see fsimage storage)
    
        # The final mask which is returned should have 
        # first channel the output of cellpose and the rest are the class channels
        # TODO test case produces img with size HxW for eval and HxWx3 for train
        with torch.no_grad():
            # get instance mask from segmentor
            instance_mask = self.segmentor.eval(img)
            # find coordinates of detected objects
            locs = find_objects(instance_mask) 
            class_mask = np.zeros(instance_mask.shape)
            # get patches centered around detected objects
            patches, _ = self.get_centered_patches(img, 
                                                   instance_mask, 
                                                   self.eval_config["classifier"]["data"]["patch_size"], 
                                                   noise_intensity=5)
            # loop over patches and create classification mask
            for idx, patch in enumerate(patches):
                patch_class = self.classifier.eval(patch)
                loc = locs[idx]
                # Assign predicted class to corresponding location in final_mask
                class_mask[loc] = patch_class.item() + 1
            # Apply mask to final_mask, retaining only regions where cellpose_mask is greater than 0
            class_mask = class_mask * (instance_mask > 0)#.long())
            final_mask = np.stack((instance_mask, class_mask)).astype(np.uint16)

        self.eval_config['z_axis'] = 0

        return final_mask

    # REMOVE? replaced by code in eval
    '''
    def get_prediction(self, input_image, cellpose_mask):
        """
        Performs object segmentation and classification on an input image using the Cellpose model and a classifier model.

        Args:
            image_path (str): The file path of the input image.
            model (CellposeModel): The Cellpose model used for object segmentation, instance segmenation mask.
         
        Returns:
            tuple: A tuple containing the cellpose_mask and final_mask, representing the segmentation masks obtained from
                the Cellpose model and the combined segmentation and classification mask, respectively.
        """

        # Obtain segmentation mask using Cellpose model

        # Find objects in the cellpose_mask
        locs = find_objects(cellpose_mask)

        # Get patches and labels based on object centroids
        patches, labels = self.get_centered_patches(input_image, cellpose_mask, int(1.5 * input_image.shape[0] // 5), noise_intensity=5)
        
        labels = torch.tensor(labels)
        labels_fit = []
        
        final_mask = torch.zeros(cellpose_mask.shape)

        with torch.no_grad():
            for i, patch in enumerate(patches):
                loc = locs[i]

                # Prepare image patch for classification
                img = torch.tensor(patch.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255
                # img = img.mean(dim=1, keepdim=True)
              
                # Perform inference using model_classifier
                logits = self.classifier(img) 
        
                _, predicted = torch.max(logits, 1)
                labels_fit.append(predicted)

                # Assign predicted class to corresponding location in final_mask
                final_mask[loc] = predicted + 1

        # Apply mask to final_mask, retaining only regions where cellpose_mask is greater than 0
        final_mask = final_mask * ((cellpose_mask > 0).long())
        
        return final_mask
    '''
    def find_max_patch_size(self, mask):

        # Find objects in the mask
        objects = find_objects(mask)

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

    def crop_centered_padded_patch(self, 
                                   x: np.ndarray, 
                                   c, 
                                   p, 
                                   mask: np.ndarray=None,
                                   noise_intensity=None) -> np.ndarray:
        """
        Crop a patch from an array `x` centered at coordinates `c` with size `p`, and apply padding if necessary.

        Args:
            x (np.ndarray): The input array from which the patch will be cropped.
            c (tuple): The coordinates (row, column, channel) at the center of the patch.
            p (tuple): The size of the patch to be cropped (height, width).

        Returns:
            np.ndarray: The cropped patch with applied padding.
        """            

        if mask.shape[0] < mask.shape[-1]:
                mask = mask.transpose(1, 2, 0)

        height, width = p  # Size of the patch

        # Calculate the boundaries of the patch
        top = c[0] - height // 2
        bottom = top + height
        
        left = c[1] - width // 2
        right = left + width

        # Crop the patch from the input array
        if mask is not None:
            mask_ = mask.max(-1) if len(mask.shape) >= 3 else mask
            # central_label = mask_[c[0], c[1]]
            central_label = mask_[c[0]][c[1]]
            # Zero out values in the patch where the mask is not equal to the central label
            # m = (mask_ != central_label) & (mask_ > 0)
            m = (mask_ != central_label) & (mask_ > 0)
            x[m] = 0
            if noise_intensity is not None:
                x[m] = np.random.normal(scale=noise_intensity, size=x[m].shape)

        patch = x[max(top, 0):min(bottom, x.shape[0]), max(left, 0):min(right, x.shape[1]), :]

        if len(c) == 3:
            patch = patch[...,c[2]]

        # Calculate the required padding amounts
        size_x, size_y = x.shape[1], x.shape[0]

        # Apply padding if necessary
        if left < 0: 
            patch = np.hstack((
                np.random.normal(scale=noise_intensity, size=(patch.shape[0], abs(left), patch.shape[2])).astype(np.uint8),
                patch))
        # Apply padding on the right side if necessary
        if right > size_x: 
            patch = np.hstack((
                patch,
                np.random.normal(scale=noise_intensity, size=(patch.shape[0], (right - size_x), patch.shape[2])).astype(np.uint8)))
        # Apply padding on the top side if necessary
        if top < 0: 
            patch = np.vstack((
                np.random.normal(scale=noise_intensity, size=(abs(top), patch.shape[1], patch.shape[2])).astype(np.uint8),
                patch))
        # Apply padding on the bottom side if necessary
        if bottom > size_y: 
            patch = np.vstack((
                patch, 
                np.random.normal(scale=noise_intensity, size=(bottom - size_y, patch.shape[1], patch.shape[2])).astype(np.uint8)))

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
        return [(int(x[0]), int(x[1])) 
                for x in center_of_mass(mask, mask, np.arange(1, mask.max() + 1))]
    
    def get_centered_patches(self,
                             img,
                             mask,
                             p_size: int,
                             noise_intensity=5,
                             mask_class=None):

        ''' 
        Extracts centered patches from the input image based on the centers of objects identified in the mask.

        Args:
            img: The input image.
            mask: The mask representing the objects in the image.
            p_size (int): The size of the patches to extract.
            noise_intensity: The intensity of noise to add to the patches.

        '''

        patches, labels = [], []
        # if image is 2D add an additional dim for channels
        if img.ndim<3: img = img[:, :, np.newaxis]
        if mask.ndim<3: mask = mask[:, :, np.newaxis]
        # compute center of mass of objects
        centers_of_mass = self.get_center_of_mass(mask)
        # Crop patches around each center of mass
        for c in centers_of_mass:
            c_x, c_y = c
            patch = self.crop_centered_padded_patch(img.copy(),
                                                   (c_x, c_y),
                                                   (p_size, p_size),
                                                   mask=mask,
                                                   noise_intensity=noise_intensity)
            patches.append(patch)
            if mask_class is not None: labels.append(mask_class[c[0]][c[1]])

        return patches, labels

    def create_patch_dataset(self, imgs, masks_classes, masks_instances):
        '''
        Splits img and masks into patches of equal size which are centered around the cells.
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

        noise_intensity = self.train_config["classifier"]["train_data"]["noise_intensity"]
        max_patch_size = self.train_config["classifier"]["train_data"]["patch_size"]
        num_classes = self.train_config["classifier"]["train_data"]["num_classes"]

        patches, labels = [], []
        for img, mask_class, mask_instance in zip(imgs,  masks_classes, masks_instances):
            # Convert to one-hot encoding
            # mask has dimension WxHxNum_of_channels
            patch, label = self.get_centered_patches(img, 
                                                        mask_instance,
                                                        self.train_config["classifier"]["train_data"]["patch_size"], 
                                                        noise_intensity=noise_intensity,
                                                        mask_class=mask_class)
            patches.extend(patch)
            labels.extend(label)
        return patches, labels
        
# class CustomSAMModel():
# # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
#     def __init__(self):
#         pass
