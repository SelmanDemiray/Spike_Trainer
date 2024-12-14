# Simple Neural Network Trainer

This repository contains a basic implementation of a neural network trainer in Python. It's designed for educational purposes to help you understand the fundamental concepts of neural networks, including forward and backward passes, loss and error calculation, and visualization of training progress.

**Key Features:**

*   Modular structure: The code is divided into logical modules (`neural_network.py`, `utilities.py`, `data_handler.py`) for easy understanding and modification.
*   Dataset Support: Handles various datasets including MNIST, CIFAR-10, Fashion MNIST, KMNIST, EMNIST, SVHN, and CIFAR-100.
*   Activation Functions: Supports sigmoid and ReLU activation functions.
*   Visualization: Visualizes weights (receptive fields) and training metrics (loss, error) to aid in understanding the network's behavior.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SelmanDemiray/Spike_Trainer](https://github.com/SelmanDemiray/Spike_Trainer)
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy matplotlib
    ```

3.  **Run the trainer:**
    ```bash
    python neural_network.py
    ```

4.  **Follow the prompts:**
    *   You'll be presented with a list of available datasets. Enter the number corresponding to your desired dataset.
    *   Choose an activation function (sigmoid or relu).

## Code Structure

*   `neural_network.py`: Main script to run the training process. Handles dataset selection, hyperparameter settings, training loop, and visualization.
*   `utilities.py`: Contains utility functions for:
    *   Activation functions (sigmoid, ReLU)
    *   Forward and backward pass calculations
    *   Loss and error calculation
    *   Visualization of weights and images
*   `data_handler.py`: Handles loading and preprocessing of different datasets.

## Understanding the Code

*   **Hyperparameters:** You can adjust hyperparameters like learning rate (`epsilon`), momentum (`alpha`), weight decay (`gamma`), number of epochs (`nepoch`), batch size (`nbatch`), and number of hidden units (`nhid`) in the `neural_network.py` file.
*   **Dataset Preprocessing:** The code includes basic preprocessing for CIFAR-10. You might need to add preprocessing steps for other datasets (e.g., normalization, standardization) in the `data_handler.py` file or within the main script.
*   **Training Loop:** The training loop in `neural_network.py` iterates through epochs and batches, performs forward and backward passes, updates weights and biases, and calculates loss and error.
*   **Visualization:** The `show_weights` function in `utilities.py` visualizes the learned weights of the first layer, and the code saves plots of the training loss and test error.

## Potential Enhancements

*   Implement additional activation functions (tanh, leaky ReLU, etc.).
*   Add support for more advanced optimization algorithms (Adam, RMSprop).
*   Incorporate regularization techniques (dropout, L1/L2 regularization).
*   Extend the code to handle deeper network architectures with more layers.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

[Choose a license, e.g., MIT License]