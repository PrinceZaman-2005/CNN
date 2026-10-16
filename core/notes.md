## Architecture

Our CNN accepts **8×8 grayscale images** and classifies them into three categories:

* Class `0`: Horizontal line
* Class `1`: Vertical line
* Class `2`: Diagonal line

These are our planned dataset labels; we'll define them in the dataset module.

### Architecture flow

```text
Input: 1 × 8 × 8
       |
       v
Conv2D: 1 → 8 channels
       |
      ReLU
       |
    MaxPool2D
       |
       v
Feature Maps: 8 × 4 × 4
       |
       v
Conv2D: 8 → 16 channels
       |
      ReLU
       |
    MaxPool2D
       |
       v
Feature Maps: 16 × 2 × 2
       |
       v
     Flatten
       |
       v
Linear: 64 → 32
       |
      ReLU
       |
       v
Linear: 32 → 3
       |
       v
Class Scores
```

## ⚡ Important Details

**1. Why `padding=1`?**

Our `3×3` convolutional filters preserve the image's height and width before pooling. This helps retain spatial dimensions at each convolution stage.

**2. Why two pooling layers?**

Each pooling layer halves the spatial dimensions:

`8×8 → 4×4 → 2×2`

**3. Why don't we use Softmax?**

The final layer produces three raw class scores, called *logits*. We'll use `nn.CrossEntropyLoss()` during training, which handles the required log-softmax operation internally.
