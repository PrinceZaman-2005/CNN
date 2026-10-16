import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    def __init__(
        self,
        model,
        learning_rate=0.001,
        batch_size=16
    ):
        self.model = model
        self.batch_size = batch_size

        # Classification loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

    def train(
        self,
        train_data,
        train_labels,
        test_data,
        test_labels,
        epochs=50
    ):
        # Convert NumPy arrays to PyTorch tensors
        train_data = torch.tensor(
            train_data,
            dtype=torch.float32
        ).unsqueeze(1)

        train_labels = torch.tensor(
            train_labels,
            dtype=torch.long
        )

        test_data = torch.tensor(
            test_data,
            dtype=torch.float32
        ).unsqueeze(1)

        test_labels = torch.tensor(
            test_labels,
            dtype=torch.long
        )

        # Create batches
        dataset = TensorDataset(
            train_data,
            train_labels
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        history = {
            "train_loss": [],
            "test_loss": [],
            "test_accuracy": []
        }

        for epoch in range(epochs):

            # Training phase
            self.model.train()

            total_loss = 0.0

            for images, labels in loader:

                # Forward pass
                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels
                )

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * images.size(0)

            train_loss = total_loss / len(dataset)

            # Evaluation phase
            test_loss, test_accuracy = self.evaluate(
                test_data,
                test_labels
            )

            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["test_accuracy"].append(test_accuracy)

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"| Train Loss: {train_loss:.4f} "
                f"| Test Loss: {test_loss:.4f} "
                f"| Test Accuracy: {test_accuracy:.2f}%"
            )

        return history

    def evaluate(self, test_data, test_labels):

        self.model.eval()

        with torch.no_grad():

            outputs = self.model(test_data)

            loss = self.criterion(
                outputs,
                test_labels
            ).item()

            predictions = outputs.argmax(dim=1)

            correct = (
                predictions == test_labels
            ).sum().item()

            accuracy = (
                correct / len(test_labels)
            ) * 100

        return loss, accuracy
