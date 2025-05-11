from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluate import evaluate_model

@pipeline(enable_cache=True)
def train_pipeline(train_path: str, test_path: str, model_name: str):
    X_full, X_test_full = ingest_data(train_path, test_path)
    X_train, X_valid, X_test, y_train, y_valid = clean_data(X_full, X_test_full)
    model = train_model(X_train, y_train, X_valid, y_valid, model_name = model_name)
    mae = evaluate_model(model, X_valid, y_valid)
