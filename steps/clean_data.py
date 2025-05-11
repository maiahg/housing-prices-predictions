import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreprocessingStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(X_full: pd.DataFrame, X_test_full: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_valid"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_valid"]
]:
    try:
        preprocessing_strategy = DataPreprocessingStrategy()
        data_cleaning = DataCleaning(X_full, X_test_full, preprocessing_strategy)
        X_train, X_valid, X_test, y_train, y_valid = data_cleaning.handle_data()
        
        return X_train, X_valid, X_test, y_train, y_valid
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise e
    