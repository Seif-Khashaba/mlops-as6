import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
import sys
import os

# -------------------------------
# 1. Define Model
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.fc = nn.Linear(1440, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 1440)
        return self.fc(x)

# -------------------------------
# 2. Training Function
# -------------------------------
def train_model(lr, epochs, batch_size):

    # Set MLflow tracking URI from environment (GitHub Secrets)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # Set experiment
    mlflow.set_experiment("Assignment5_Seif_Khashaba")

    # Start MLflow run
    with mlflow.start_run() as run:

        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # -------------------------------
        # Log parameters
        # -------------------------------
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.set_tag("student_id", "Seif_Khashaba")

        # -------------------------------
        # Load Data (MNIST)
        # -------------------------------
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True, transform=transform),
            batch_size=batch_size,
            shuffle=True
        )

        # -------------------------------
        # Model setup
        # -------------------------------
        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        # -------------------------------
        # Training Loop
        # -------------------------------
        model.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0
            correct = 0

            for batch_idx, (data, target) in enumerate(train_loader):

                optimizer.zero_grad()

                output = model(data)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            avg_loss = total_loss / len(train_loader)
            accuracy = correct / len(train_loader.dataset)

            # Log metrics
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)

            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

        # -------------------------------
        # Log model
        # -------------------------------
        mlflow.pytorch.log_model(model, "model")

        # -------------------------------
        # Save Run ID for pipeline
        # -------------------------------
        with open("model_info.txt", "w") as f:
            f.write(run_id)

        print("Saved run_id to model_info.txt")


# -------------------------------
# 3. Entry Point
# -------------------------------
if __name__ == "__main__":

    learning_rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    num_epochs    = int(sys.argv[2])   if len(sys.argv) > 2 else 5
    batch_size    = int(sys.argv[3])   if len(sys.argv) > 3 else 64

    print(f"Starting run with: LR={learning_rate}, Epochs={num_epochs}, Batch={batch_size}")

    train_model(
        lr=learning_rate,
        epochs=num_epochs,
        batch_size=batch_size
    )