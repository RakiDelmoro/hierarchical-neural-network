import numpy as np

import numpy as np

class PredictiveCodingNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        self.W0 = np.random.randn(input_size, hidden_size) * 0.1  # Input to hidden
        self.W1 = np.random.randn(hidden_size, output_size) * 0.1 # Hidden to output
    
    def forward(self, x):
        # Initial forward pass to set initial activities
        mu0 = x.reshape(-1, 1)
        mu1 = self.W0.T @ mu0  # Transpose for prediction
        mu2 = self.W1.T @ mu1
        return mu0, mu1, mu2
    
    def infer_activities(self, mu0, mu1, mu2, y, inference_rate=0.1, steps=10):
        y = y.reshape(-1, 1)
        for _ in range(steps):
            # Compute prediction errors
            epsilon0 = mu0 - self.W0 @ mu1
            epsilon1 = mu1 - self.W1 @ mu2
            epsilon2 = mu2 - y

            # Compute gradients for activities
            d_mu1 = epsilon1 + self.W0.T @ epsilon0
            d_mu2 = epsilon2 + self.W1.T @ epsilon1

            # Update activities
            mu1 += inference_rate * d_mu1
            mu2 += inference_rate * d_mu2
        return mu1, mu2

    def update_weights(self, mu0, mu1, mu2, learning_rate=0.01):
        # Compute errors with settled activities
        epsilon0 = mu0 - self.W0 @ mu1
        epsilon1 = mu1 - self.W1 @ mu2

        # Update weights (Hebbian-like updates)
        self.W0 -= learning_rate * epsilon0 @ mu1.T
        self.W1 -= learning_rate * epsilon1 @ mu2.T

    def train(self, X, y, epochs=1000, inference_rate=0.1, learning_rate=0.01, inference_steps=20):
        for epoch in range(epochs):
            total_error = 0
            for xi, target in zip(X, y):
                # Initial forward pass
                mu0, mu1, mu2 = self.forward(xi)
                
                # Inference phase to settle activities
                mu1, mu2 = self.infer_activities(mu0, mu1, mu2, target, inference_rate, inference_steps)
                
                # Compute total error for monitoring
                epsilon2 = mu2 - target.reshape(-1, 1)
                total_error += 0.5 * np.sum(epsilon2**2)
                
                # Update weights
                self.update_weights(mu0, mu1, mu2, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Error: {total_error / len(X):.4f}")

# Example: XOR Problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train the network
network = PredictiveCodingNetwork(input_size=2, hidden_size=2, output_size=1)
network.train(X, y, epochs=1000, learning_rate=0.01, inference_steps=6)