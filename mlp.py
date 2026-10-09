import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Initializes the neural network with random weights and zero biases.
        """
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        # W1: weights from input to hidden layer
        self.W1 = np.random.randn(input_size, hidden_size) * 1.0
        self.b1 = np.zeros((1, hidden_size))
        
        # W2: weights from hidden to output layer
        self.W2 = np.random.randn(hidden_size, output_size) * 1.0
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function for backpropagation"""
        return x * (1 - x)
    
    def forward(self, X):
        """
        Forward propagation pass.
        Returns the final prediction.
        """
        # Layer 1 (Hidden Layer)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Layer 2 (Output Layer)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """
        Backward propagation pass (calculates gradients and updates weights).
        """
        m = y.shape[0] # Number of training examples
        
        # Calculate loss derivative with respect to output
        error = output - y
        
        # Backpropagate to hidden layer
        dZ2 = error * self.sigmoid_derivative(output)
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Backpropagate to input layer
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights and biases using gradient descent
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
    def train(self, X, y, epochs=10000):
        """
        Trains the neural network over a specified number of epochs.
        """
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y, output)
            
            # Optional: Print loss every 1000 epochs to monitor training
            if (epoch % 1000) == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch} | Mean Squared Error (Loss): {loss:.4f}")

if __name__ == "__main__":
    # The XOR Problem Dataset
    # XOR is a classic non-linear problem that requires at least one hidden layer to solve.
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    y_train = np.array([
        [0],
        [1],
        [1],
        [0]
    ])
    
    print("Initializing Multi-Layer Perceptron (MLP) Neural Network...")
    print("Network Architecture: 2 Inputs -> 4 Hidden Neurons -> 1 Output")
    
    # 2 input features, 4 hidden nodes, 1 output node
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
    
    print("\nStarting Training on XOR Dataset (10,000 Epochs)...")
    nn.train(X_train, y_train, epochs=10000)
    
    print("\nTraining Complete. Testing Predictions:")
    predictions = nn.forward(X_train)
    
    for i in range(len(X_train)):
        print(f"Input: {X_train[i]} | Target: {y_train[i][0]} | Predicted: {predictions[i][0]:.4f} -> Rounded: {round(predictions[i][0])}")
