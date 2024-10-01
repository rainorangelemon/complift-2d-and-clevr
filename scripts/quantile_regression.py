import torch
import numpy as np
import wandb

import torch.nn as nn
import torch.optim as optim

# Define the neural network model
class QuantileRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(QuantileRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Quantile loss function
def quantile_loss(preds, target, quantile):
    errors = target - preds
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    return torch.mean(loss)

# Generate some synthetic data
np.random.seed(0)
torch.manual_seed(0)
x = np.random.rand(100000, 1)
y = 2 * x + 1 + np.random.randn(100000, 1) * 0.1

x_train = torch.tensor(x, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

# Hyperparameters
input_dim = x_train.shape[1]
hidden_dim = 10
learning_rate = 0.01
num_epochs = 1000
quantile = 0.9

# Initialize wandb
wandb.init(project="quantile-regression", config={
    "learning_rate": learning_rate,
    "epochs": num_epochs,
    "quantile": quantile,
    "hidden_dim": hidden_dim
})

# Initialize the model, loss function and optimizer
model = QuantileRegressionModel(input_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(x_train)
    loss = quantile_loss(outputs, y_train, quantile)

    loss.backward()
    optimizer.step()

    # Log the loss and quantile to wandb
    wandb.log({"epoch": epoch + 1, "loss": loss.item(), "quantile": quantile})

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    test_input = torch.tensor([[0.5]], dtype=torch.float32)
    prediction = model(test_input)
    print(f'Prediction for input 0.5: {prediction.item():.4f}')

# Finish the wandb run
wandb.finish()