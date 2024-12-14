import numpy as np
import os
import urllib.request
import gzip
import struct
import requests
import tarfile

def load_mnist(path):
    """
    Loads the MNIST dataset.

    Parameters:
    path (str): Path to the directory where MNIST data is stored.

    Returns:
    tuple: Training and testing images and labels.
    """
    def read_images(file_path):
        with open(file_path, 'rb') as f:
            magic_number, n_images, n_rows, n_columns = struct.unpack('>IIII', f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(n_images, n_rows * n_columns)
            images = images.T / 255.0
        return images

    def read_labels(file_path):
        with open(file_path, 'rb') as f:
            magic_number, n_labels = struct.unpack('>II', f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
            one_hot_labels = np.zeros((10, n_labels))
            one_hot_labels[labels, np.arange(n_labels)] = 1.0
        return one_hot_labels

    train_images = read_images(os.path.join(path, 'train-images-idx3-ubyte'))
    train_labels = read_labels(os.path.join(path, 'train-labels-idx1-ubyte'))

    test_images = read_images(os.path.join(path, 't10k-images-idx3-ubyte'))
    test_labels = read_labels(os.path.join(path, 't10k-labels-idx1-ubyte'))

    return train_images, train_labels, test_images, test_labels

def load_cifar10(path):
    """
    Loads the CIFAR-10 dataset.

    Parameters:
    path (str): Path to the directory where CIFAR-10 data is stored.

    Returns:
    tuple: Training and testing images and labels.
    """
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    train_data = []
    train_labels = []
    for i in range(1, 6):
        data_dict = unpickle(os.path.join(path, f'data_batch_{i}'))
        train_data.append(data_dict[b'data'])
        train_labels.extend(data_dict[b'labels'])

    train_data = np.concatenate(train_data, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = np.array(train_labels)

    test_dict = unpickle(os.path.join(path, 'test_batch'))
    test_data = test_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = np.array(test_dict[b'labels'])

    # Convert labels to one-hot encoding
    train_labels_one_hot = np.zeros((train_labels.size, 10))
    train_labels_one_hot[np.arange(train_labels.size), train_labels] = 1
    test_labels_one_hot = np.zeros((test_labels.size, 10))
    test_labels_one_hot[np.arange(test_labels.size), test_labels] = 1

    return train_data, train_labels_one_hot, test_data, test_labels_one_hot

def download_and_extract(url, path):
    """
    Downloads and extracts a compressed dataset using requests.

    Parameters:
    url (str): URL of the dataset.
    path (str): Path to the directory where the dataset should be saved.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = os.path.join(path, url.split('/')[-1])
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    with open(file_name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    if file_name.endswith('.gz'):
        with gzip.open(file_name, 'rb') as f_in:
            with open(file_name[:-3], 'wb') as f_out:
                f_out.write(f_in.read())
    elif file_name.endswith('.tar.gz'):
        with tarfile.open(file_name, 'r:gz') as tar:
            tar.extractall(path)

def get_dataset(dataset_name):
    """
    Retrieves the specified dataset, downloading it if necessary.

    Parameters:
    dataset_name (str): Name of the dataset ('mnist', 'cifar10', 'fashion_mnist', 'kmnist', 'emnist', 'svhn', 'cifar100').

    Returns:
    tuple: Training and testing images and labels.
    """
    if dataset_name == 'mnist':
        path = '../mnist_data'
        if not os.path.exists(os.path.join(path, 'train-images-idx3-ubyte')):
            download_and_extract('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', path)
            download_and_extract('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', path)
            download_and_extract('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', path)
            download_and_extract('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', path)
        return load_mnist(path)
    elif dataset_name == 'cifar10':
        path = '../cifar10_data'
        if not os.path.exists(os.path.join(path, 'data_batch_1')):
            download_and_extract('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', path)
        return load_cifar10(path)
    elif dataset_name == 'fashion_mnist':
        path = '../fashion_mnist_data'
        if not os.path.exists(os.path.join(path, 'train-images-idx3-ubyte')):
            download_and_extract('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz', path)
            download_and_extract('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz', path)
            download_and_extract('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz', path)
            download_and_extract('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz', path)
        return load_mnist(path)  # You can reuse the load_mnist function for Fashion MNIST
    elif dataset_name == 'kmnist':
        path = '../kmnist_data'
        if not os.path.exists(os.path.join(path, 'train-images-idx3-ubyte')):
            download_and_extract('http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz', path)
            download_and_extract('http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz', path)
            download_and_extract('http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz', path)
            download_and_extract('http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz', path)
        return load_mnist(path)  # You can reuse the load_mnist function for KMNIST
    elif dataset_name == 'emnist':
        # Add EMNIST download and loading logic here
        pass
    elif dataset_name == 'svhn':
        # Add SVHN download and loading logic here
        pass
    elif dataset_name == 'cifar100':
        # Add CIFAR-100 download and loading logic here
        pass
    else:
        raise ValueError('Invalid dataset name.')