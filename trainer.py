# neural_network.py

import numpy as np
import matplotlib.pyplot as plt
from data_handler import get_dataset
from utilities import (
    sigmoid, 
    relu, 
    tanh, 
    softmax, 
    forward_pass, 
    backward_pass, 
    calculate_loss, 
    calculate_error, 
    show_images, 
    show_weights,
    save_weights,
    load_weights,
    convert_weights_for_snn,
    normalize_weights
)

def save_trained_weights(W0, W1, auto_save=False):
    if auto_save:
        filepath = 'weights.npz'
    else:
        filepath = input("Enter the filepath to save the weights (e.g., 'weights.npz'): ")
    save_weights(filepath, W0, W1)
    print(f"Weights saved to {filepath}")

def load_trained_weights():
    filepath = input("Enter the filepath to load the weights (e.g., 'weights.npz'): ")
    W0, W1 = load_weights(filepath)
    print(f"Weights loaded from {filepath}")
    return W0, W1

def save_snn_weights(W0, W1, auto_save=False):
    W0_snn, W1_snn = convert_weights_for_snn(W0, W1)
    if auto_save:
        filepath = 'snn_weights.npz'
    else:
        filepath = input("Enter the filepath to save the SNN weights (e.g., 'snn_weights.npz'): ")
    save_weights(filepath, W0_snn, W1_snn)
    print(f"SNN weights saved to {filepath}")

def save_normalized_snn_weights(W0, W1, auto_save=False):
    W0_norm, W1_norm = normalize_weights(W0, W1)
    W0_snn, W1_snn = convert_weights_for_snn(W0_norm, W1_norm)
    if auto_save:
        filepath = 'normalized_snn_weights.npz'
    else:
        filepath = input("Enter the filepath to save the normalized SNN weights (e.g., 'normalized_snn_weights.npz'): ")
    save_weights(filepath, W0_snn, W1_snn)
    print(f"Normalized SNN weights saved to {filepath}")

# Main script
if __name__ == "__main__":
    print("Available datasets:")
    datasets = {
        1: 'mnist',
        2: 'cifar10',
        3: 'fashion_mnist',
        4: 'kmnist',
        5: 'emnist',
        6: 'svhn',
        7: 'cifar100',
        # ... add more datasets here
    }
    for i, dataset_name in datasets.items():
        print(f"{i}. {dataset_name}")

    while True:
        try:
            dataset_choice = int(input("Enter the number of the dataset to use (default: 1): ") or 1)
            if dataset_choice in datasets:
                break
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    dataset_name = datasets[dataset_choice]

    train_images, train_labels, test_images, test_labels = get_dataset(dataset_name)

    # Preprocess images if necessary
    if dataset_name == 'cifar10':
        train_images = train_images.reshape(train_images.shape[0], -1).T / 255.0
        test_images = test_images.reshape(test_images.shape[0], -1).T / 255.0
    # Add more dataset-specific preprocessing here if needed

    # Hyperparameters
    epsilon = float(input("Enter learning rate (epsilon, default: 0.01): ") or 0.01)
    alpha = float(input("Enter momentum (alpha, default: 0.1): ") or 0.1)
    gamma = float(input("Enter weight decay (gamma, default: 0.001): ") or 0.001)
    nepoch = int(input("Enter number of epochs (nepoch, default: 10): ") or 10)
    nbatch = int(input("Enter number of batches (nbatch, default: 10): ") or 10)
    nhid = int(input("Enter number of hidden units (nhid, default: 10): ") or 10)
    sigma = float(input("Enter standard deviation for weight initialization (sigma, default: 0.1): ") or 0.1)

    # Choose activation function
    while True:
        activation_choice = input("Enter activation function ('sigmoid', 'relu', or 'tanh', default: 'sigmoid'): ").lower() or 'sigmoid'
        if activation_choice in ('sigmoid', 'relu', 'tanh'):
            break
        else:
            print("Invalid choice. Please enter 'sigmoid', 'relu', or 'tanh'.")

    activation_function = sigmoid if activation_choice == 'sigmoid' else relu if activation_choice == 'relu' else tanh

    input_size = train_images.shape[0]
    W0 = np.random.randn(nhid, input_size) * sigma
    W1 = np.random.randn(10, nhid) * sigma
    b0 = np.zeros((nhid, 1))
    b1 = np.zeros((10, 1))

    delta_W0 = np.zeros(W0.shape)
    delta_W1 = np.zeros(W1.shape)
    delta_b0 = np.zeros(b0.shape)
    delta_b1 = np.zeros(b1.shape)

    batch_size = train_images.shape[1] // nbatch

    batch_loss = np.zeros(nbatch)
    train_loss = np.zeros(nepoch + 1)
    test_error = np.zeros(nepoch + 1)

    # Choose optimizer
    while True:
        optimizer_choice = input("Enter optimizer ('sgd', 'adam', or 'rmsprop', default: 'adam'): ").lower() or 'adam'
        if optimizer_choice in ('sgd', 'adam', 'rmsprop'):
            break
        else:
            print("Invalid choice. Please enter 'sgd', 'adam', or 'rmsprop'.")

    optimizer_params = {}
    if optimizer_choice == 'adam':
        optimizer_params = {
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 0.001,
            't': 1,
            'mW0': np.zeros_like(W0),
            'mW1': np.zeros_like(W1),
            'mb0': np.zeros_like(b0),
            'mb1': np.zeros_like(b1),
            'vW0': np.zeros_like(W0),
            'vW1': np.zeros_like(W1),
            'vb0': np.zeros_like(b0),
            'vb1': np.zeros_like(b1)
        }
    elif optimizer_choice == 'rmsprop':
        optimizer_params = {
            'beta': 0.9,
            'epsilon': 0.001,
            'vW0': np.zeros_like(W0),
            'vW1': np.zeros_like(W1),
            'vb0': np.zeros_like(b0),
            'vb1': np.zeros_like(b1)
        }

    print('|||-----------------------------------------------------------------')
    print('Beginning neural network training:')
    print('|||-----------------------------------------------------------------')

    train_r0, train_r1 = forward_pass(train_images, W0, W1, b0, b1, activation_function)
    test_r0, test_r1 = forward_pass(test_images, W0, W1, b0, b1, activation_function)
    train_loss[0] = calculate_loss(train_r1, train_labels) / nbatch
    test_error[0] = calculate_error(test_r1, test_labels)
    print(f'Pre-training loss = {train_loss[0]:.3f}, test error = {test_error[0] * 100:.1f}%.')

    auto_save = input("Do you want to automatically save weights after training? (yes/no, default: no): ").lower() == 'yes'

    while True:
        action = input("Enter 'train' to train the network, 'load' to load weights, 'save_snn' to save weights for SNN, 'save_normalized_snn' to save normalized weights for SNN, or 'exit' to quit: ").lower()
        if action == 'train':
            for epoch in range(1, nepoch + 1):
                for batch in range(nbatch):
                    batch_images = train_images[:, batch * batch_size:(batch + 1) * batch_size]
                    batch_labels = train_labels[:, batch * batch_size:(batch + 1) * batch_size]

                    # Use the chosen activation function in forward and backward pass
                    r0, r1 = forward_pass(batch_images, W0, W1, b0, b1, activation_function)

                    dL_by_dW0, dL_by_dW1, dL_by_db0, dL_by_db1 = backward_pass(
                        batch_images, batch_labels, r0, r1, W0, W1, b0, b1, activation_function, optimizer_choice, optimizer_params
                    )

                    if optimizer_choice == 'sgd':
                        delta_W0 = -epsilon * dL_by_dW0 + alpha * delta_W0 - gamma * W0
                        delta_W1 = -epsilon * dL_by_dW1 + alpha * delta_W1 - gamma * W1
                        delta_b0 = -epsilon * dL_by_db0 + alpha * delta_b0 - gamma * b0
                        delta_b1 = -epsilon * dL_by_db1 + alpha * delta_b1 - gamma * b1

                        W0 += delta_W0
                        W1 += delta_W1
                        b0 += delta_b0
                        b1 += delta_b1
                    elif optimizer_choice == 'adam':
                        optimizer_params['t'] += 1

                    batch_loss[batch] = calculate_loss(r1, batch_labels)

                train_loss[epoch] = np.mean(batch_loss)
                test_r0, test_r1 = forward_pass(test_images, W0, W1, b0, b1, activation_function)
                test_error[epoch] = calculate_error(test_r1, test_labels)

                print(f'Epoch {epoch}, loss = {train_loss[epoch]:.3f}, test error = {test_error[epoch] * 100:.1f}%.')

            save_trained_weights(W0, W1, auto_save)
        elif action == 'load':
            W0, W1 = load_trained_weights()
        elif action == 'save_snn':
            save_snn_weights(W0, W1, auto_save)
        elif action == 'save_normalized_snn':
            save_normalized_snn_weights(W0, W1, auto_save)
        elif action == 'exit':
            break
        else:
            print("Invalid choice. Please enter 'train', 'load', 'save_snn', 'save_normalized_snn', or 'exit'.")

    show_weights(W0, dataset_name)
    plt.savefig('weights.png', dpi=300)

    plt.figure(2)
    plt.plot(range(nepoch + 1), train_loss, 'k-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (AU)')
    plt.title('Train loss')
    plt.grid(False)
    plt.savefig('loss.png', dpi=300)

    plt.figure(3)
    plt.plot(range(nepoch + 1), test_error * 100.0, 'r-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.title('Test error')
    plt.grid(False)
    plt.savefig('error.png', dpi=300)

    print('|||-----------------------------------------------------------------')
    print('Finished training. Figures have been saved in the current folder.')
    print('|||-----------------------------------------------------------------')