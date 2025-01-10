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

def tanh(x):
    """
    Calculates the tanh function on each element of a matrix.

    Parameters:
    x (numpy.ndarray or float): Input array or scalar.

    Returns:
    numpy.ndarray or float: Output after applying the tanh function element-wise.
    """
    return np.tanh(x)

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


def backward_pass(images, labels, r0, r1, W0, W1, b0, b1, activation_function, optimizer, optimizer_params):
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
    optimizer (str): Optimizer to use ('adam' or other).
    optimizer_params (dict): Parameters for the optimizer.

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
    if activation_function == sigmoid:
        dr0_by_dx0 = r0 * (1 - r0)
    elif activation_function == relu:
        dr0_by_dx0 = (r0 > 0).astype(float)
    elif activation_function == tanh:
        dr0_by_dx0 = 1 - r0 ** 2

    # Gradient of the loss with respect to the hidden layer input
    dL_by_dx0 = dL_by_dr0 * dr0_by_dx0  

    # Gradient of the hidden layer input with respect to the hidden layer weights
    dx0_by_dW0 = images  

    # Gradient of the loss with respect to the hidden layer weights and biases
    dL_by_dW0 = np.dot(dL_by_dx0, dx0_by_dW0.T)  
    dL_by_db0 = np.sum(dL_by_dx0, axis=1, keepdims=True)  

    if optimizer == 'adam':
        beta1, beta2, epsilon, t = optimizer_params['beta1'], optimizer_params['beta2'], optimizer_params['epsilon'], optimizer_params['t']
        
        # Update biased first moment estimate
        optimizer_params['mW0'] = beta1 * optimizer_params['mW0'] + (1 - beta1) * dL_by_dW0
        optimizer_params['mW1'] = beta1 * optimizer_params['mW1'] + (1 - beta1) * dL_by_dW1
        optimizer_params['mb0'] = beta1 * optimizer_params['mb0'] + (1 - beta1) * dL_by_db0
        optimizer_params['mb1'] = beta1 * optimizer_params['mb1'] + (1 - beta1) * dL_by_db1

        # Update biased second raw moment estimate
        optimizer_params['vW0'] = beta2 * optimizer_params['vW0'] + (1 - beta2) * (dL_by_dW0 ** 2)
        optimizer_params['vW1'] = beta2 * optimizer_params['vW1'] + (1 - beta2) * (dL_by_dW1 ** 2)
        optimizer_params['vb0'] = beta2 * optimizer_params['vb0'] + (1 - beta2) * (dL_by_db0 ** 2)
        optimizer_params['vb1'] = beta2 * optimizer_params['vb1'] + (1 - beta2) * (dL_by_db1 ** 2)

        # Compute bias-corrected first moment estimate
        mW0_hat = optimizer_params['mW0'] / (1 - beta1 ** t)
        mW1_hat = optimizer_params['mW1'] / (1 - beta1 ** t)
        mb0_hat = optimizer_params['mb0'] / (1 - beta1 ** t)
        mb1_hat = optimizer_params['mb1'] / (1 - beta1 ** t)

        # Compute bias-corrected second raw moment estimate
        vW0_hat = optimizer_params['vW0'] / (1 - beta2 ** t)
        vW1_hat = optimizer_params['vW1'] / (1 - beta2 ** t)
        vb0_hat = optimizer_params['vb0'] / (1 - beta2 ** t)
        vb1_hat = optimizer_params['vb1'] / (1 - beta2 ** t)

        # Update weights and biases
        W0 -= epsilon * mW0_hat / (np.sqrt(vW0_hat) + 1e-8)
        W1 -= epsilon * mW1_hat / (np.sqrt(vW1_hat) + 1e-8)
        b0 -= epsilon * mb0_hat / (np.sqrt(vb0_hat) + 1e-8)
        b1 -= epsilon * mb1_hat / (np.sqrt(vb1_hat) + 1e-8)
    elif optimizer == 'rmsprop':
        beta, epsilon = optimizer_params['beta'], optimizer_params['epsilon']
        
        # Update running average of squared gradients
        optimizer_params['vW0'] = beta * optimizer_params['vW0'] + (1 - beta) * (dL_by_dW0 ** 2)
        optimizer_params['vW1'] = beta * optimizer_params['vW1'] + (1 - beta) * (dL_by_dW1 ** 2)
        optimizer_params['vb0'] = beta * optimizer_params['vb0'] + (1 - beta) * (dL_by_db0 ** 2)
        optimizer_params['vb1'] = beta * optimizer_params['vb1'] + (1 - beta) * (dL_by_db1 ** 2)

        # Update weights and biases
        W0 -= epsilon * dL_by_dW0 / (np.sqrt(optimizer_params['vW0']) + 1e-8)
        W1 -= epsilon * dL_by_dW1 / (np.sqrt(optimizer_params['vW1']) + 1e-8)
        b0 -= epsilon * dL_by_db0 / (np.sqrt(optimizer_params['vb0']) + 1e-8)
        b1 -= epsilon * dL_by_db1 / (np.sqrt(optimizer_params['vb1']) + 1e-8)

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

def save_weights(filepath, W0, W1):
    """
    Saves the weights to a file.

    Parameters:
    filepath (str): Path to the file where weights will be saved.
    W0 (numpy.ndarray): Weights matrix for hidden layer.
    W1 (numpy.ndarray): Weights matrix for output layer.
    """
    np.savez(filepath, W0=W0, W1=W1)

def load_weights(filepath):
    """
    Loads the weights from a file.

    Parameters:
    filepath (str): Path to the file from which weights will be loaded.

    Returns:
    tuple: Loaded weights matrices (W0, W1).
    """
    data = np.load(filepath)
    return data['W0'], data['W1']

def convert_weights_for_snn(W0, W1):
    """
    Converts the weights to a format compatible with the SNN.

    Parameters:
    W0 (numpy.ndarray): Weights matrix for hidden layer.
    W1 (numpy.ndarray): Weights matrix for output layer.

    Returns:
    tuple: Converted weights matrices (W0_snn, W1_snn).
    """
    W0_snn = W0.T  # Transpose to match SNN input format
    W1_snn = W1.T  # Transpose to match SNN input format
    return W0_snn, W1_snn

def normalize_weights(W0, W1):
    """
    Normalizes the weights to ensure they are suitable for the SNN.

    Parameters:
    W0 (numpy.ndarray): Weights matrix for hidden layer.
    W1 (numpy.ndarray): Weights matrix for output layer.

    Returns:
    tuple: Normalized weights matrices (W0_norm, W1_norm).
    """
    W0_norm = W0 / np.linalg.norm(W0, axis=1, keepdims=True)
    W1_norm = W1 / np.linalg.norm(W1, axis=1, keepdims=True)
    return W0_norm, W1_norm