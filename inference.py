import mlflow

model = mlflow.pyfunc.load_model(
    model_uri=(
        "models:/BreastCancerRF/latest"
    )
)

print(
    "Model loaded successfully"
)