import os
import tempfile

import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from mlflow.models import infer_signature


# -------------------------------------
# Experiment
# -------------------------------------

mlflow.set_experiment(
    "Breast Cancer Classification"
)

# -------------------------------------
# Dataset
# -------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------
# Hyperparameter Search Space
# -------------------------------------

configs = [
    {
        "n_estimators": 50,
        "max_depth": 5
    },
    {
        "n_estimators": 100,
        "max_depth": 10
    },
    {
        "n_estimators": 200,
        "max_depth": None
    }
]

best_accuracy = 0
best_run_id = None

# -------------------------------------
# Training Loop
# -------------------------------------

for config in configs:

    with mlflow.start_run():

        # -----------------------------
        # Tags
        # -----------------------------

        mlflow.set_tags({
            "project": "breast-cancer",
            "team": "mlops-learning",
            "framework": "sklearn"
        })

        # -----------------------------
        # Parameters
        # -----------------------------

        mlflow.log_params(config)

        # -----------------------------
        # Model
        # -----------------------------

        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=42
        )

        model.fit(X_train, y_train)

        # -----------------------------
        # Predictions
        # -----------------------------

        predictions = model.predict(X_test)

        # -----------------------------
        # Metrics
        # -----------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # -----------------------------
        # Confusion Matrix Artifact
        # -----------------------------

        cm = confusion_matrix(
            y_test,
            predictions
        )

        plt.figure(figsize=(5, 4))
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.colorbar()

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        temp_dir = tempfile.mkdtemp()

        cm_path = os.path.join(
            temp_dir,
            "confusion_matrix.png"
        )

        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(
            cm_path,
            artifact_path="plots"
        )

        # -----------------------------
        # Feature Importance Artifact
        # -----------------------------

        importance_df = pd.DataFrame({
            "feature":
                X.columns,
            "importance":
                model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False
        )

        importance_csv = os.path.join(
            temp_dir,
            "feature_importance.csv"
        )

        importance_df.to_csv(
            importance_csv,
            index=False
        )

        mlflow.log_artifact(
            importance_csv,
            artifact_path="feature_importance"
        )

        # -----------------------------
        # Signature
        # -----------------------------

        signature = infer_signature(
            X_train,
            model.predict(X_train)
        )

        # -----------------------------
        # Input Example
        # -----------------------------

        input_example = X_train.head(5)

        # -----------------------------
        # Model Logging
        # -----------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="BreastCancerRF"
        )

        # -----------------------------
        # Best Model Tracking
        # -----------------------------

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_run_id = mlflow.active_run().info.run_id

        print(
            f"Run completed | "
            f"Accuracy={accuracy:.4f}"
        )

print("\nBest Run:")
print(best_run_id)
print("Best Accuracy:", best_accuracy)