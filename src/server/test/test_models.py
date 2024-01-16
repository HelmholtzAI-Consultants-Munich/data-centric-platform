import numpy as np
import pytest
# from models import CellClassifierShallowModel 
from sklearn.exceptions import NotFittedError

import dcp_server.models as models
from dcp_server.utils import read_config
from synthetic_dataset import get_synthetic_dataset


"""
self.classifier = CellClassifierShallowModel(self.model_config,
                                             self.train_config,
                                             self.eval_config)
"""

def test_eval_rf_not_fitted():

    model_config = read_config('model', config_path='test/test_config_RF.cfg')
    train_config = read_config('train', config_path='test/test_config_RF.cfg')
    eval_config = read_config('eval', config_path='test/test_config_RF.cfg')
    
    model_rf = models.CellClassifierShallowModel(model_config,train_config,eval_config)

    X_test = np.array([[1, 2, 3]]) 
    # if we don't fit the model then the model returns zeros
    assert np.all(model_rf.eval(X_test)== np.zeros(X_test.shape))

def test_update_configs():

    model_config = read_config('model', config_path='test/test_config_RF.cfg')
    train_config = read_config('train', config_path='test/test_config_RF.cfg')
    eval_config = read_config('eval', config_path='test/test_config_RF.cfg')
    
    model = models.CustomCellposeModel(model_config,train_config,eval_config)

    new_train_config = {"param1": "value1"}
    new_eval_config = {"param2": "value2"}

    model.update_configs(new_train_config, new_eval_config)

    assert model.train_config == new_train_config
    assert model.eval_config == new_eval_config



