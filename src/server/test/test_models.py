import pytest
import numpy as np

import dcp_server.models as models
from dcp_server.models.classifiers import FeatureClassifier
from dcp_server.utils.helpers import read_config

def test_eval_rf_not_fitted():
    """
    Tests the evaluation of a random forest model that has not been fitted.
    """

    model_config = read_config('model', config_path='test/configs/test_config_Inst2MultiSeg_RF.yaml')
    data_config = read_config('data', config_path='test/configs/test_config_Inst2MultiSeg_RF.yaml')
    train_config = read_config('train', config_path='test/configs/test_config_Inst2MultiSeg_RF.yaml')
    eval_config = read_config('eval', config_path='test/configs/test_config_Inst2MultiSeg_RF.yaml')
    
    model_rf = FeatureClassifier("Random Forest", model_config, data_config, train_config, eval_config)

    X_test = np.array([[1, 2, 3]]) 
    # if we don't fit the model then the model returns zeros
    assert np.all(model_rf.eval(X_test)== np.zeros(X_test.shape))

def test_update_configs():
    """
    Tests the update of model training and evaluation configurations.
    """

    model_config = read_config('model', config_path='test/configs/test_config_Inst2MultiSeg_RF.yaml')
    data_config = read_config('data', config_path='test/configs/test_config_Inst2MultiSeg_RF.yaml')
    train_config = read_config('train', config_path='test/configs/test_config_Inst2MultiSeg_RF.yaml')
    eval_config = read_config('eval', config_path='test/configs/test_config_Inst2MultiSeg_RF.yaml')
    
    model = models.CustomCellpose("Cellpose", model_config, data_config, train_config, eval_config)

    new_train_config = {"param1": "value1"}
    new_eval_config = {"param2": "value2"}

    model.update_configs(new_train_config, new_eval_config)

    assert model.train_config == new_train_config
    assert model.eval_config == new_eval_config



