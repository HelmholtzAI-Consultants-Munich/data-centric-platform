import pytest
import numpy as np

import dcp_server.models as models
from dcp_server.utils import read_config

def test_eval_rf_not_fitted():
    """
    Tests the evaluation of a random forest model that has not been fitted.
    """

    model_config = read_config('model', config_path='test/configs/test_config_RF.cfg')
    train_config = read_config('train', config_path='test/configs/test_config_RF.cfg')
    eval_config = read_config('eval', config_path='test/configs/test_config_RF.cfg')
    
    model_rf = models.CellClassifierShallowModel(model_config,train_config,eval_config)

    X_test = np.array([[1, 2, 3]]) 
    # if we don't fit the model then the model returns zeros
    assert np.all(model_rf.eval(X_test)== np.zeros(X_test.shape))

def test_update_configs():
    """
    Tests the update of model training and evaluation configurations.
    """

    model_config = read_config('model', config_path='test/configs/test_config_RF.cfg')
    train_config = read_config('train', config_path='test/configs/test_config_RF.cfg')
    eval_config = read_config('eval', config_path='test/configs/test_config_RF.cfg')
    
    model = models.CustomCellposeModel(model_config,train_config,eval_config, "Cellpose")

    new_train_config = {"param1": "value1"}
    new_eval_config = {"param2": "value2"}

    model.update_configs(new_train_config, new_eval_config)

    assert model.train_config == new_train_config
    assert model.eval_config == new_eval_config



