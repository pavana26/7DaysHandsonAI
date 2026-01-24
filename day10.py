# Import required libraries
import numpy as np


#  Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

# Mean squared error loss function 
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

#  Basic Neural Network class
class BasicNeuralNetwork:
    def __init__(self,input_size,hidden_size,output_size):
        self.weights_input_hidden = np.random.rand(input_size,hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size,output_size)    
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)

    # Forward pass
    def forward(self,X):
        self.hidden_layer_activation = np.dot(X,self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)
        self.output_layer_activation = np.dot(self.hidden_layer_output,self.weights_hidden_output) + self.bias_output
        output = sigmoid(self.output_layer_activation)
        return output
    
    # Backward pass and weights update
    def backward(self,X,y,output,learning_rate):
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * sigmoid_derivative(self.hidden_layer_output)

        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_layer_delta) * learning_rate

        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0) * learning_rate

    # Train the neural network
    def train(self,X,y,epochs,learning_rate):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            # Backward pass and weights
            self.backward(X,y,output,learning_rate)
            if (epoch+1) % 1000 == 0:
                loss = mean_squared_error(y,output)
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

# Example usage
X =np.array([[0,0],[0,1],[1,0],[1,1]])
y =np.array([[0],[1],[1],[0]])  # XOR problem
nn = BasicNeuralNetwork(input_size=2,hidden_size=2,output_size=1)
nn.train(X,y,epochs=10000,learning_rate=0.1)    

print("\nTest the trained neural network:")
for i in range(len(X)):
    output = nn.forward(X[i].reshape(1,-1))
    print(f"Input: {X[i]}, Predicted Output: {output[0][0]:.4f}',Actual Output: {y[i]}")