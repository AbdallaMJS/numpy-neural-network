# NumPy Neural Network (Multi-Layer Perceptron)

A lightweight, fully functional Multi-Layer Perceptron (MLP) artificial neural network built entirely from scratch using only `NumPy`. This project avoids high-level frameworks like TensorFlow or PyTorch to explicitly demonstrate the mathematics behind deep learning.

## 🌟 Key Features

- **Built from Scratch:** No external machine learning libraries. Every mathematical operation for the neural network is explicitly defined using matrix algebra via NumPy.
- **Forward Propagation:** Calculates activations through the network layers using dot products and a Sigmoid activation function.
- **Backpropagation Engine:** Implements the chain rule of calculus to compute loss gradients backwards through the network, updating weights and biases.
- **Gradient Descent Optimization:** Adjusts network parameters iteratively to minimize Mean Squared Error (MSE).
- **Non-Linear Problem Solving:** Effectively solves the classic XOR problem, demonstrating that the hidden layer architecture successfully captures non-linear relationships.

## 🛠️ Technologies Used

- **Language:** Python
- **Core Library:** `NumPy` (for matrix multiplication and array operations)

## 💡 How It Works (The Math)

The network consists of an Input Layer (2 nodes), a Hidden Layer (4 nodes), and an Output Layer (1 node). 

1. **Forward Pass:** The input `X` is multiplied by the weight matrix `W1`, and bias `b1` is added. The result passes through a non-linear Sigmoid activation function to become the hidden layer output `a1`. This process is repeated for the output layer.
2. **Error Calculation:** The network compares its prediction against the actual target using Mean Squared Error.
3. **Backward Pass (Backpropagation):** The network calculates the derivative of the error with respect to the output, and then applies the chain rule to find the gradient of the error with respect to `W2`, `b2`, `W1`, and `b1`.
4. **Parameter Update:** Using Gradient Descent, the weights and biases are updated by moving slightly in the opposite direction of the gradient (scaled by the learning rate).

## 🚀 Setup and Execution

To run this project, you only need Python and NumPy.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AbdallaMJS/numpy-neural-network.git
   cd numpy-neural-network
   ```

2. **Install NumPy:**
   ```bash
   pip install numpy
   ```

3. **Run the network:**
   ```bash
   python mlp.py
   ```
   The script will train the network on the XOR dataset and output its progressively decreasing loss, followed by its final predictions.

## 🧠 Educational Value & MBZUAI Relevance

Understanding how neural networks operate "under the hood" is essential for advanced AI research. By manually coding backpropagation and gradient descent, this project demonstrates a concrete grasp of linear algebra, multivariate calculus, and machine learning fundamentals. It proves an ability to manipulate matrices and understand mathematical optimization, core prerequisites for the MBZUAI Undergraduate AI program.

---
*Developed by Abdalla M.J.S. Alblooshi to showcase foundational knowledge in deep learning architecture and mathematical optimization.*
