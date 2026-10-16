# ⬜ Convolutional Neural Network (CNN)
A simple Convolutional Neural Network (CNN) built with PyTorch to explore image classification, convolutional filters, and spatial feature extraction.

This project demonstrates how CNNs learn to recognize patterns in images and how modern deep-learning frameworks simplify building and training neural networks.

## ⚡ Introduction

A **Convolutional Neural Network (CNN)** is a neural network architecture commonly used for processing grid-like data, such as images.

Unlike a basic fully connected neural network, a CNN uses convolutional filters to examine local regions of an input. These filters learn to detect useful patterns, such as edges, lines, and other spatial features.

### Architecture

```text
       Input Image
            |
            v
      ┌───────────┐
      │ Convolution│
      └───────────┘
            |
            v
          ReLU
            |
            v
      ┌───────────┐
      │  Pooling  │
      └───────────┘
            |
            v
      ┌───────────┐
      │ Convolution│
      └───────────┘
            |
            v
          ReLU
            |
            v
        Flatten
            |
            v
      ┌───────────┐
      │  Linear   │
      │   Layer   │
      └───────────┘
            |
            v
      Class Scores
            |
            v
     Predicted Class
```

### How Does It Work?

1. **Input:** The network receives an image represented as numerical data.
2. **Convolution:** Learnable filters scan local regions of the image to extract spatial features.
3. **Activation:** ReLU introduces non-linearity, allowing the network to learn more complex patterns.
4. **Pooling:** Reduces the spatial dimensions of feature maps, helping produce a more compact representation.
5. **Feature extraction:** Successive layers combine simpler features into more complex patterns.
6. **Flattening:** Converts the extracted feature maps into a vector for classification.
7. **Classification:** Fully connected layers produce scores for the possible classes.
8. **Learning:** A loss function measures the prediction error, and backpropagation updates the network's parameters.

The objective is to learn spatial features that help the model distinguish between different image categories.

> CNN can learn useful image features, but its performance depends on the dataset, architecture, training procedure, and evaluation method.

## Why Use PyTorch?

Building a CNN from scratch requires implementing convolution operations, calculating gradients, and managing parameter updates.

PyTorch provides tools that simplify these tasks.

| Task                          | With PyTorch                              |
| ----------------------------- | ----------------------------------------- |
| Define convolutional layers   | `torch.nn.Conv2d`                         |
| Apply activation functions    | `torch.nn.ReLU`                           |
| Reduce feature-map dimensions | `torch.nn.MaxPool2d`                      |
| Define fully connected layers | `torch.nn.Linear`                         |
| Calculate classification loss | `torch.nn.CrossEntropyLoss`               |
| Calculate gradients           | Automatic differentiation with `autograd` |
| Update parameters             | `torch.optim`                             |
| Process image data            | PyTorch tensors                           |

For example, a convolutional layer can be defined using:

```python
nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3
)
```

This layer accepts a single-channel input and learns eight convolutional filters, each with a spatial kernel size of `3 × 3`.

During training, PyTorch calculates gradients for the learnable parameters and provides optimizers that use those gradients to update them.

This allows us to focus more on **understanding convolution, designing architectures, and conducting experiments** rather than implementing every low-level operation ourselves.

**Note:** This project can be trained using the CPU version of PyTorch, making it suitable for environments such as GitHub Codespaces without requiring a GPU.

## Trade-Off: Spatial Feature Extraction vs. Generality

CNNs introduce useful architectural assumptions for image processing, but these assumptions also involve trade-offs.

### Advantages

* **Local feature extraction:** Convolutional filters examine nearby input values to identify spatial patterns.
* **Parameter sharing:** The same filter is applied across different spatial locations, reducing the number of parameters compared with many fully connected alternatives.
* **Hierarchical features:** Deeper layers can combine simpler features into more complex representations.
* **Translation-related robustness:** Convolution and pooling can help the model recognize patterns at different locations, although this does not guarantee complete translation invariance.
* **Efficient image processing:** Convolutions can exploit the spatial structure of images instead of connecting every input pixel independently to every hidden unit.

### Trade-Offs

* **Local receptive fields:** A convolutional layer initially focuses on local regions, so capturing distant relationships may require deeper layers or other mechanisms.
* **Spatial information loss:** Pooling reduces spatial resolution and can discard useful details.
* **Inductive bias:** CNNs are designed around local spatial structure, which may not be the best assumption for every type of data.
* **Dataset dependency:** A model trained on a limited set of images may not generalize well to unfamiliar patterns.
* **Resource requirements:** Larger images, wider layers, and deeper architectures can increase memory consumption and computation time.

These trade-offs do not make CNNs inherently better or worse than fully connected networks. The appropriate architecture depends on the data and the problem being solved.

For image classification, convolutional layers provide a useful way to learn spatial features. For other tasks, recurrent networks, Transformers, or different architectures may be more appropriate.

**The goal is not simply to use convolution. It is to understand why spatial structure matters and how the architecture exploits it.**

## What This Project Explores

* Convolutional neural network architecture
* Convolutional filters and feature maps
* ReLU activation
* Pooling and spatial dimensionality reduction
* Feature extraction and classification
* Cross-entropy loss
* Backpropagation and optimizer-based training

## Technologies

* Python
* PyTorch
* NumPy
* Matplotlib

The exact classification performance depends on the dataset, model configuration, and training process.

## Learning Objective

This project continues the series by exploring a neural network architecture designed to learn spatial features from images.

Earlier projects explored individual neurons, backpropagation, energy-based learning, and latent representations. Here, we examine how convolutional layers use local connectivity and shared filters to learn patterns in image data.

The aim is to understand both sides:

* **What the model learns from images.**
* **How convolutional layers extract and combine spatial features.**

## About the Series

**10 Days of Modeling AI for Beginners**

A hands-on series exploring AI concepts through small, understandable projects.

The objective is to learn by building, experimenting, and examining the mechanisms behind AI—not simply by importing a library and accepting its output.

### What's Next?

**Day 8 — Recurrent Neural Network (RNN)**

Moving from spatial patterns in images to sequential information, where the order of the input matters.
