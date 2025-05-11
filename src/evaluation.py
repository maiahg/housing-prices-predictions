import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error

class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


class MAE(Evaluation):
    """
    Evaluation strategy that uses Mean Absolute Error (MAE)
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mae: float
        """
        try:
            logging.info("Entered the calculate_score method of the MAE class")
            mae = mean_absolute_error(y_true, y_pred)
            logging.info("The mean absolute error value is: " + str(mae))
            return mae
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the MAE class. Exception message:  "
                + str(e)
            )
            raise e