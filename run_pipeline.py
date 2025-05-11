from pipelines.train_pipeline import train_pipeline

if __name__ == "__main__":
    # Run the pipeline
    train_pipeline(
        train_path="data/train.csv",
        test_path="data/test.csv",
        model_name="RandomForestModel"
    )