# mytorch

**mytorch** is a nano-scale deep learning library built from scratch
using NumPy. Inspired by the modular design of **PyTorch**, it provides
a transparent, "under-the-hood" look at how neural networks are
constructed and trained.

------------------------------------------------------------------------

## 🚀 Features

-   **Modular Layer API** --- Easily stack layers with custom
    dimensions.
-   **Simple Training** --- Use the built-in `.train()` method to handle
    backpropagation and weight updates in one go.
-   **Fast Inference** --- Use `.feed_forward()` to get predictions from
    your model.
-   **Modern Python Tooling** --- Uses `pyproject.toml` for easy
    installation and dependency management.

------------------------------------------------------------------------

## 🛠 Installation

Install locally in **editable mode**:

``` bash
git clone https://github.com/yourusername/mytorch.git
cd mytorch
pip install -e .
```

------------------------------------------------------------------------

## 💻 Quick Start

### 1. Define Your Network

``` python
import mytorch as mt

# Define layers: mt.Layer(output_neurons, input_neurons, activation)
l1 = mt.Layer(128, 784, mt.activation.relu)
l2 = mt.Layer(10, 128, mt.activation.softmax)

# Create the network with a loss function
nn = mt.NeuralNetwork([l1, l2], mt.loss.cce_loss)
```

### 2. Training and Prediction

The deep learning workflow is simplified into two primary methods:

-   `nn.train(x, y, learning_rate)`\
    Computes the loss, runs backpropagation, updates weights, and
    returns the loss.
-   `nn.feed_forward(x)`\
    Returns predictions from the final layer.

``` python
# Training a single sample
loss = nn.train(x_train, y_train, learning_rate=0.01)

# Predicting on new data
predictions = nn.feed_forward(x_test)
```

------------------------------------------------------------------------

## 🏗 Project Structure

    mytorch/
     ├── activation.py   # ReLU, Softmax, and other activations
     ├── loss.py         # Loss functions (e.g., Categorical Cross-Entropy)
     ├── layer.py        # Definition of Layer of the neural network
     └── network.py      # Definition of the NeuralNetwork itself

------------------------------------------------------------------------

## 👤 Author

**Mukesh Aryal**\
📧 Email: mukeshnuwakot@gmail.com

