## How the training module works?

| Step | Operation          | Purpose                                  |
| ---- | ------------------ | ---------------------------------------- |
| 1    | Convert to tensors | Prepare image and label data for PyTorch |
| 2    | `unsqueeze(1)`     | Add the grayscale channel dimension      |
| 3    | `DataLoader`       | Divide the dataset into batches          |
| 4    | Forward pass       | Predict the image classes                |
| 5    | `CrossEntropyLoss` | Measure classification error             |
| 6    | `loss.backward()`  | Calculate gradients                      |
| 7    | `optimizer.step()` | Update model parameters                  |
| 8    | Evaluation         | Measure performance on test images       |

### Why `unsqueeze(1)`?

Our dataset produces images with shape:

```text
(batch_size, 8, 8)
```

But `nn.Conv2d` expects:

```text
(batch_size, channels, height, width)
```

Therefore:

```python
train_data = train_data.unsqueeze(1)
```

Changes the shape to:

```text
(batch_size, 1, 8, 8)
```

The `1` represents a single grayscale channel.
