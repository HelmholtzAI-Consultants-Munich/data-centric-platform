from cellpose import models, utils
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

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

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

    #def train (self, x, y):
    ## TODO should call forward repeatedly and perform the entire train loop
    
    #def eval(self, x):
    ## TODO should call forward once, model is in eval mode, and return predicted masks

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.final_conv(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        
        return x


# class CustomSAMModel():
# # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
#     def __init__(self):
#         pass


class CellposePatchCNN():

    """Cellpose & patches of cells and then cnn to classify each patch
    """
    
    def __init__(self,  model_config, train_config, eval_config ):

        # Initialize the cellpose model
        self.train_config = train_config
        self.eval_config = eval_config

        self.classifier = CellFullyConvClassifier()
        self.segmentor = CustomCellposeModel(model_config, train_config, eval_config)

    def train(self, imgs, masks):

        # masks should have first channel as a cellpose mask and
        # all other layers corresponds to the classes
        ## TODO: take care of the images and masks preparation -> this step isn't for now @Mariia
        self.segmentor.train(imgs, masks)
        ## TODO call create_patches (adjust create_patch_dataset function)
        ## to prepare imgs and masks for training CNN
        ## TODO call self.classifier.train(imgs, masks)

    def eval(self, img, **eval_config):
        pass
        ## TODO implement the eval pipeline, i.e. first call self.segmentor.eval, then split again into patches
        ## using resulting seg and then call self.classifier.eval. The final mask which is returned should have 
        ## first channel the output of cellpose and the rest are the class channels

    def create_patch_dataset(self, imgs, masks, black_bg:bool, include_mask:bool):
        '''
        TODO: Split img and masks into patches of equal size which are centered around the cells.
        The algorithm should first run through all images to find the max cell size, and use
        the max cell size to define the patch size. All patches and masks should then be returned
        in the same foramt as imgs and masks (same type, i.e. check if tensor or np.array and same 
        convention of dims, e.g.  CxHxW)
        Args:
            imgs ():
            masks ():
            black_bg (bool): Flag indicating whether to use a black background for patches.
            include_mask (bool): Flag indicating whether to include the mask along with patches.
        '''

        for img, msk in zip(imgs, masks):
            
            for channel in range(num_of_channels):

                loc = find_objects(msk[channel])
                msk_patches = get_patches(loc, img, msk[channel], black_bg=black_bg, include_mask=include_mask)
                save_patches(msk_patches,channel, save_imgs_path)





