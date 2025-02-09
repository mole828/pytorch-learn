import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Define the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature, one output feature

    def forward(self, x):
        return self.linear(x)

def f(x):
    return 2*x + 1

# 2. Generate some synthetic data
X = np.array([[i] for i in range(10)], dtype=np.float32)
y = np.array([[f(i)] for i in range(10)], dtype=np.float32) # y = 2x + 1

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

# 3. Instantiate the model
model = LinearRegressionModel()

# 4. Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# 5. Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    
    # Backward and optimize
    optimizer.zero_grad()  # zero the gradient buffers
    loss.backward()
    optimizer.step()  # Updates the weights

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. Evaluate the model
print('Predicted y values:')
predicted = model(X_train).detach().numpy()
for i in range(len(X)):
    print(f'Input = {X[i][0]:.2f}, Predicted = {predicted[i][0]:.2f}, Actual = {y[i][0]:.2f}')

# 7. Print the learned parameters
w = model.linear.weight.item()
b = model.linear.bias.item()
print(f'Learned parameters: w = {w:.2f}, b = {b:.2f}')
