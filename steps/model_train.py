import logging

import pandas as pd
from zenml import step

from src.model_dev import RandomForestModel, XGBoostModel, HyperparameterTuner
from sklearn.base import RegressorMixin

MODEL_REGISTRY = {
    "RandomForestModel": RandomForestModel,
    "XGBoostModel": XGBoostModel,
}

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_name: str = "RandomForestModel",  # Default to RandomForestModel
) -> RegressorMixin:
    try:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model {model_name} not found")
            
        model_class = MODEL_REGISTRY[model_name]
        model = model_class()
        trained_model = model.train(X_train, y_train)
        return trained_model
        
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise e
