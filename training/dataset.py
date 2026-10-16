import numpy as np


def create_dataset(
    num_samples=100,
    image_size=8,
    test_ratio=0.2
):
    """
    Create a labeled binary image dataset.

    Classes:
        0: Horizontal line
        1: Vertical line
        2: Main diagonal

    Returns:
        train_data: Training images
        train_labels: Training labels
        test_data: Testing images
        test_labels: Testing labels
    """

    images = []
    labels = []

    for _ in range(num_samples):

        image = np.zeros(
            (image_size, image_size),
            dtype=np.float32
        )

        pattern = np.random.randint(0, 3)

        if pattern == 0:
            # Horizontal line
            row = np.random.randint(0, image_size)
            image[row, :] = 1

        elif pattern == 1:
            # Vertical line
            column = np.random.randint(0, image_size)
            image[:, column] = 1

        else:
            # Main diagonal
            np.fill_diagonal(image, 1)

        images.append(image)
        labels.append(pattern)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Shuffle the dataset before splitting
    indices = np.random.permutation(num_samples)

    images = images[indices]
    labels = labels[indices]

    # Split into training and testing sets
    test_size = int(num_samples * test_ratio)

    test_data = images[:test_size]
    test_labels = labels[:test_size]

    train_data = images[test_size:]
    train_labels = labels[test_size:]

    return (
        train_data,
        train_labels,
        test_data,
        test_labels
    )
