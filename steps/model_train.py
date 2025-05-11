import logging
import pandas as pd
from zenml import step

@step
def train_model(df: pd.DataFrame) -> pd.DataFrame:
    """Train a model on the data"""
    logging.info(f"Training model on the data")
    return df
    