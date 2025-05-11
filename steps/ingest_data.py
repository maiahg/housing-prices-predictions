import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingest data from the file_path
    """
    def __init__(self, train_path: str, test_path: str):
        """
        Args:
            train_path (str): path to the training data
            test_path (str): path to the test data
        """
        self.train_path = train_path
        self.test_path = test_path

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ingest data from the file_path
        """
        logging.info(f"Ingesting data from {self.train_path} and {self.test_path}")
        return pd.read_csv(self.train_path, index_col='Id'), pd.read_csv(self.test_path, index_col='Id')

@step
def ingest_data(train_path: str, test_path: str) -> pd.DataFrame:
    """
    Ingest data from the file_path
    
    Args:
        train_path: path to the training data
        test_path: path to the test data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(train_path, test_path)
        df_train, df_test = ingest_data.get_data()
        logging.info(f"Data ingested successfully from {train_path} and {test_path}")
        return df_train, df_test
    except Exception as e:
        logging.error(f"Error ingesting data from {train_path} and {test_path}: {e}")
        raise e
