from abc import ABC, abstractmethod

import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model on the given data.

        Args:
            X_train: Training data
            y_train: Target data
        """
        pass

    @abstractmethod
    def optimize(self, trial, X_train, y_train, X_valid, y_valid):
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            X_train: Training data
            y_train: Target data
            X_valid: Validation data
            y_valid: Validation target
        """
        pass


class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        reg = RandomForestRegressor(**kwargs)
        reg.fit(X_train, y_train)
        return reg

    def optimize(self, trial, X_train, y_train, X_valid, y_valid):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = self.train(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        return reg.score(X_valid, y_valid)

class XGBoostModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        reg = xgb.XGBRegressor(**kwargs)
        reg.fit(X_train, y_train)
        return reg

    def optimize(self, trial, X_train, y_train, X_valid, y_valid):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 30)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)
        reg = self.train(X_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        return reg.score(X_valid, y_valid)    

class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__(self, model, X_train, y_train, X_valid, y_valid):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.X_train, self.y_train, self.X_valid, self.y_valid), n_trials=n_trials)
        return study.best_trial.params
