import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        logging.info(f"Ingesting data from {self.train_path} and {self.test_path}")
        return pd.read_csv(self.train_path, index_col='Id'), pd.read_csv(self.test_path, index_col='Id')

@step
def ingest_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        ingest_data = IngestData(train_path, test_path)
        X_full, X_test_full = ingest_data.get_data()
        logging.info(f"Data ingested successfully from {train_path} and {test_path}")
        return X_full, X_test_full
    except Exception as e:
        logging.error(f"Error ingesting data from {train_path} and {test_path}: {e}")
        raise e
