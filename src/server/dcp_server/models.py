from cellpose import models
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator


class CustomCellposeModel(models.CellposeModel):

    def __init__(self, model_config, train_config, eval_config, **kwargs):
        
        # Initialize the cellpose model
        super().__init__(**model_config)
        self.train_config = train_config
        self.eval_config = eval_config
        

    def eval(self, img, z_axis):

        return super().eval(x=img, z_axis=z_axis, **self.eval_config)
    
    def train(self, imgs, masks):
        super().train(train_data=imgs, train_labels=masks, **self.train_config)
        

# class CustomSAMModel():
# # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
#     def __init__(self):
#         pass



