import pytest
import numpy as np

import dcp_server.models as models
from dcp_server.models.classifiers import FeatureClassifier
from dcp_server.utils.helpers import read_config


def test_eval_rf_not_fitted():
    """
    Tests the evaluation of a random forest model that has not been fitted.
    """

    model_config = read_config(
        "model", config_path="test/configs/test_config_Inst2MultiSeg_RF.yaml"
    )
    data_config = read_config(
        "data", config_path="test/configs/test_config_Inst2MultiSeg_RF.yaml"
    )
    eval_config = read_config(
        "eval", config_path="test/configs/test_config_Inst2MultiSeg_RF.yaml"
    )
    model_rf = FeatureClassifier(
        "Random Forest", model_config, data_config, eval_config
    )

    X_test = np.array([[1, 2, 3]])
    # if we don't fit the model then the model returns zeros
    assert np.all(model_rf.eval(X_test) == np.zeros(X_test.shape))

