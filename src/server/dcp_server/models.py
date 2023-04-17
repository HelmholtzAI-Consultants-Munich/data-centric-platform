# Here, we will define the models class which will enable us to have more flexibility on the model parameters
# Curently, we decide only on the model type when definining the bentoml runner in service.py
from cellpose import models
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator


class CustomCellposeModel(models.CellposeModel):

    def __init__(self, model_type, **kwargs):
        
        # Initialize the cellpose model
        super().__init__(model_type=model_type, **kwargs)

    def eval(self, img, z_axis, **kwargs):
        '''
        **kwargs added to pass any other argument that original eval function (from cellposemodel) accepts 
        '''
        return super().eval(x=img, z_axis=z_axis, **kwargs)
    

    def train(self, imgs, masks, n_epochs, channels, save_path, **kwargs):
        '''
        **kwargs added to pass any other argument that original train function (from cellposemodel) accepts 
        '''
        super().train(train_data=imgs, train_labels=masks, n_epochs=n_epochs, channels=channels, 
                         save_path=save_path, **kwargs)
        


# class CustomSAMModel():
# # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
#     def __init__(self):
#         pass



