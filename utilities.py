# utilities.py

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Calculates the sigmoid function on each element of a matrix.

    Parameters:
    x (numpy.ndarray or float): Input array or scalar.

    Returns:
    numpy.ndarray or float: Output after applying the sigmoid function element-wise.
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """
    Calculates the ReLU function on each element of a matrix.

    Parameters:
    x (numpy.ndarray or float): Input array or scalar.

    Returns:
    numpy.ndarray or float: Output after applying the ReLU function element-wise.
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Calculates the softmax function for each column of a matrix.

    Parameters:
    x (numpy.ndarray): Input matrix.

    Returns:
    numpy.ndarray: Output after applying the softmax function to each column.
    """
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Numerical stability
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def forward_pass(images, W0, W1, b0, b1, activation_function):
    """
    Run a forward pass through the network.

    Parameters:
    images (numpy.ndarray): Input images matrix. Shape depends on the dataset.
    W0 (numpy.ndarray): Weights matrix for hidden layer. Shape depends on the dataset and nhid.
    W1 (numpy.ndarray): Weights matrix for output layer. Shape depends on nhid.
    b0 (numpy.ndarray): Bias vector for hidden layer. Shape depends on nhid.
    b1 (numpy.ndarray): Bias vector for output layer.
    activation_function (function): Activation function to use (sigmoid or relu).

    Returns:
    r0 (numpy.ndarray): Hidden layer activity matrix.
    r1 (numpy.ndarray): Output layer activity matrix.
    """
    r0 = activation_function(np.dot(W0, images) + b0)
    r1 = softmax(np.dot(W1, r0) + b1)  # Softmax for output layer
    return r0, r1


def backward_pass(images, labels, r0, r1, W0, W1, b0, b1, activation_function):
    """
    Run a backward pass through the network.

    Parameters:
    images (numpy.ndarray): Input images matrix. Shape depends on the dataset.
    labels (numpy.ndarray): Target output matrix.
    r0 (numpy.ndarray): Hidden layer activity matrix.
    r1 (numpy.ndarray): Output layer activity matrix.
    W0 (numpy.ndarray): Weights matrix for hidden layer.
    W1 (numpy.ndarray): Weights matrix for output layer.
    b0 (numpy.ndarray): Bias vector for hidden layer.
    b1 (numpy.ndarray): Bias vector for output layer.
    activation_function (function): Activation function used (sigmoid or relu).

    Returns:
    dL_by_dW0 (numpy.ndarray): Gradient of the loss with respect to W0.
    dL_by_dW1 (numpy.ndarray): Gradient of the loss with respect to W1.
    dL_by_db0 (numpy.ndarray): Gradient of the loss with respect to b0.
    dL_by_db1 (numpy.ndarray): Gradient of the loss with respect to b1.
    """

    # Gradient of the loss with respect to the output layer activity
    dL_by_dr1 = r1 - labels 

    # Gradient of the output layer activity with respect to the output layer input
    dr1_by_dx1 = r1 * (1 - r1)  

    # Gradient of the loss with respect to the output layer input
    dL_by_dx1 = dL_by_dr1 * dr1_by_dx1  

    # Gradient of the output layer input with respect to the output weights
    dx1_by_dW1 = r0  

    # Gradient of the loss with respect to the output weights and biases
    dL_by_dW1 = np.dot(dL_by_dx1, dx1_by_dW1.T)  
    dL_by_db1 = np.sum(dL_by_dx1, axis=1, keepdims=True)  

    # Gradient of the output layer input with respect to the hidden layer activity
    dx1_by_dr0 = W1  

    # Gradient of the loss with respect to the hidden layer activity
    dL_by_dr0 = np.dot(dx1_by_dr0.T, dL_by_dx1)  

    # Gradient of the hidden layer activity with respect to the hidden layer input
    dr0_by_dx0 = r0 * (1 - r0)  

    # Gradient of the loss with respect to the hidden layer input
    dL_by_dx0 = dL_by_dr0 * dr0_by_dx0  

    # Gradient of the hidden layer input with respect to the hidden layer weights
    dx0_by_dW0 = images  

    # Gradient of the loss with respect to the hidden layer weights and biases
    dL_by_dW0 = np.dot(dL_by_dx0, dx0_by_dW0.T)  
    dL_by_db0 = np.sum(dL_by_dx0, axis=1, keepdims=True)  

    return dL_by_dW0, dL_by_dW1, dL_by_db0, dL_by_db1

def calculate_loss(r1, labels):
    """
    Calculates the cross-entropy loss.

    Parameters:
    r1 (numpy.ndarray): Output matrix (probabilities).
    labels (numpy.ndarray): True labels matrix (one-hot encoded).

    Returns:
    L (float): Cross-entropy loss.
    """
    L = -np.sum(labels * np.log(r1 + 1e-8))  # Add small value for numerical stability
    return L

def calculate_error(r1, labels):
    """
    Calculates the classification error.

    Parameters:
    r1 (numpy.ndarray): Output matrix.
    labels (numpy.ndarray): True labels matrix.

    Returns:
    E (float): Classification error rate.
    """
    predictions = np.argmax(r1, axis=0)
    true_labels = np.argmax(labels, axis=0)
    E = np.mean(predictions != true_labels)
    return E

def show_images(images, labels, dataset_name):
    """
    Displays sample images from the dataset.

    Parameters:
    images (numpy.ndarray): Images matrix.
    labels (numpy.ndarray): One-hot encoded labels matrix.
    dataset_name (str): Name of the dataset.
    """
    plt.figure(figsize=(10, 10))
    plt.gray()

    selection = np.zeros((10, 10), dtype=int)
    for cat in range(10):
        categories = np.argmax(labels, axis=0)
        this_category = np.where(categories == cat)[0]
        selection[cat, :] = np.random.choice(this_category, 10, replace=False)

    for r in range(10):
        for c in range(10):
            imagenumber = r * 10 + c + 1
            plt.subplot(10, 10, imagenumber)
            if dataset_name in ('mnist', 'fashion_mnist', 'kmnist'):
                plt.imshow(images[:, selection[r, c]].reshape(28, 28).T, aspect='equal')
            elif dataset_name == 'cifar10':
                plt.imshow(images[selection[r, c], :, :, :], aspect='equal')
            # Add more dataset-specific image display logic here if needed
            plt.axis('off')

    plt.suptitle('Sample images', fontsize=16)
    plt.show()

def show_weights(weights, dataset_name):
    """
    Displays the receptive fields for MNIST, Fashion MNIST, and KMNIST.

    Parameters:
    weights (numpy.ndarray): Weights matrix.
    dataset_name (str): Name of the dataset.
    """
    if dataset_name in ('mnist', 'fashion_mnist', 'kmnist'):
        plt.figure()
        plt.gray()

        n_fields = weights.shape[0]
        n_sqrt = np.sqrt(n_fields)
        n_rows = int(np.round(n_sqrt))
        n_cols = int(np.ceil(n_fields / n_rows))

        for r in range(n_rows):
            for c in range(n_cols):
                field_number = r * n_cols + c + 1
                if field_number <= n_fields:
                    plt.subplot(n_rows, n_cols, field_number)
                    plt.imshow(weights[field_number - 1, :].reshape(28, 28).T, aspect='equal')
                    plt.axis('off')

        plt.suptitle('Receptive fields', fontsize=16)
        plt.show()
    elif dataset_name == 'cifar10':
        # For CIFAR-10, we'll skip showing the weights as they are not easily interpretable
        pass
    # Add more dataset-specific weight display logic here if needed