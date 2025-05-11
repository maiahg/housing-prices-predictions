import logging
import pandas as pd
from zenml import step

@step
def evaluate_model(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate a model on the data"""
    logging.info(f"Evaluating model on the data")
    return df
    
    