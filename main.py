import random

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from core.cnn import CNN
from training.dataset import create_dataset
from training.train import Trainer


def set_seed(seed=42):
    """Make experiments more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def visualize_predictions(
    model,
    test_data,
    test_labels,
    num_images=8
):
    """Save a comparison of test images and their predictions."""

    model.eval()

    # Prepare images for Conv2d
    images = torch.tensor(
        test_data,
        dtype=torch.float32
    ).unsqueeze(1)

    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1).numpy()

    class_names = [
        "Horizontal",
        "Vertical",
        "Diagonal"
    ]

    count = min(num_images, len(test_data))

    fig, axes = plt.subplots(
        2,
        (count + 1) // 2,
        figsize=(12, 6)
    )

    axes = np.array(axes).reshape(-1)

    for i in range(count):
        axes[i].imshow(
            test_data[i],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        actual = class_names[test_labels[i]]
        predicted = class_names[predictions[i]]

        axes[i].set_title(
            f"Actual: {actual}\nPredicted: {predicted}"
        )

        axes[i].axis("off")

    for i in range(count, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    plt.close()

    print("\nVisualization saved to predictions.png")


def main():
    set_seed(62)

    # 1. Generate the dataset
    train_data, train_labels, test_data, test_labels = (
        create_dataset(
            num_samples=1000,
            image_size=8,
            test_ratio=0.2
        )
    )

    print("Dataset generated!")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")

    # 2. Initialize the CNN
    model = CNN(num_classes=3)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(model)
    print(f"\nTrainable parameters: {total_parameters}")

    # 3. Train the model
    trainer = Trainer(
        model=model,
        learning_rate=0.001,
        batch_size=16
    )

    print("\nStarting training...\n")

    history = trainer.train(
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        epochs=20
    )

    # 4. Display final metrics
    print("\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final test loss: {history['test_loss'][-1]:.4f}")
    print(f"Final test accuracy: {history['test_accuracy'][-1]:.2f}%")

    # 5. Save prediction visualization
    visualize_predictions(
        model,
        test_data,
        test_labels
    )

if __name__ == "__main__":
    main()
