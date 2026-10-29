
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Configure MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Experiment_5")

# Load dataset
data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

results = []

# Train and track models
for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("model_name", name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", accuracy)
        results.append({
            "model": name,
            "accuracy": accuracy,
            "run_id": run.info.run_id
        })

        results.append({
            "model": name,
            "accuracy": accuracy,
            "run_id": run.info.run_id
        })

# Compare experiments
df = pd.DataFrame(results)
print("\nExperiment Comparison:")
print(df.to_string(index=False))

# Select the best model
best = max(results, key=lambda x: x["accuracy"])
print("\nBest Model:", best["model"])
print("Best Accuracy:", round(best["accuracy"], 4))

# Register the best model
model_uri = f"runs:/{best['run_id']}/model"
registered = mlflow.register_model(
    model_uri=model_uri,
    name="Best_Iris_Model"
)

print("Registered Model:", registered.name)
print("Model Version:", registered.version)

df = pd.DataFrame(results)
print(df)

best = max(results, key=lambda x: x["accuracy"])
print("Best Model:", best["model"])
print("Best Accuracy:", best["accuracy"])