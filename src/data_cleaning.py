import logging
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """
    
    @abstractmethod
    def handle_data(self, X_full: pd.DataFrame, X_test_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Handle the data"""
        pass

class DataPreprocessingStrategy(DataStrategy):
    def handle_data(self, X_full: pd.DataFrame, X_test_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            # Remove rows with missing target, separate target from predictors
            X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
            y = X_full.SalePrice
            X_full.drop(['SalePrice'], axis=1, inplace=True)
            
            # Break off validation set from training data
            X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)
            
            # Select categorical columns with relatively low cardinality
            categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]
            
            # Select numerical columns
            numerical_cols = [cname for cname in X_train_full.columns if 
                    X_train_full[cname].dtype in ['int64', 'float64']]
            
            # Keep selected columns only
            my_cols = categorical_cols + numerical_cols
            X_train = X_train_full[my_cols].copy()
            X_valid = X_valid_full[my_cols].copy()
            X_test = X_test_full[my_cols].copy()
            
            # Preprocessing for numerical data
            numerical_transformer = SimpleImputer(strategy='constant')
            
            # Preprocessing for categorical data
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            # Bundle preprocessing for numerical and categorical data
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])
            
            # Preprocess training data
            X_train_processed = pd.DataFrame(preprocessor.fit_transform(X_train))
            
            # Preprocess validation data
            X_valid_processed = pd.DataFrame(preprocessor.transform(X_valid))
            
            # Preprocess test data
            X_test_processed = pd.DataFrame(preprocessor.transform(X_test))
            
            return X_train_processed, X_valid_processed, X_test_processed, y_train, y_valid
        
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise e

class DataCleaning:

    def __init__(self, X_full: pd.DataFrame, X_test_full: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.X_full = X_full
        self.X_test_full = X_test_full
        self.strategy = strategy

    def handle_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.X_full, self.X_test_full)    