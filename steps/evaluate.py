import logging

import mlflow
import pandas as pd
from src.evaluation import MAE
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step()
def evaluate_model(
    model: RegressorMixin, X_valid: pd.DataFrame, y_valid: pd.Series
) -> Annotated[float, "mae"]:
    """
    Args:
        model: RegressorMixin
        X_valid: pd.DataFrame
        y_valid: pd.Series
    Returns:
        mae: float
    """
    try:
        prediction = model.predict(X_valid)

        # Using the MAE class for mean absolute error calculation
        mae_class = MAE()
        mae = mae_class.calculate_score(y_valid, prediction)
        mlflow.log_metric("mae", mae)

        return mae
    except Exception as e:
        logging.error(e)
        raise e
