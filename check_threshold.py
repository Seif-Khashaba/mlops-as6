import mlflow
import sys

THRESHOLD = 0.85

# Read run id
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

# Connect to MLflow server
mlflow.set_tracking_uri("YOUR_MLFLOW_URI")

# Get run
run = mlflow.get_run(run_id)

# Get accuracy
metrics = run.data.metrics
accuracy = metrics.get("accuracy", 0)

print(f"Accuracy: {accuracy}")

if accuracy < THRESHOLD:
    print("Model failed threshold ❌")
    sys.exit(1)
else:
    print("Model passed threshold ✅")