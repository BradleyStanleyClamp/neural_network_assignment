import neural_network as nn
import torch


# Hyperparameters
epochs = 16
learning_rate = 3e-3
optimizer = "adam"
batch_size = 64

# Model
linear_model = nn.Linear_model()

# Using training class to setup and run training
train_network = nn.Train_network(
    "mnist", learning_rate, optimizer, batch_size, epochs, linear_model
)

# Training the model
train_network.train_model(live_plot_bool=True)

# TODO: save the model


# Evaluate model
# test_loader = train_network.test_loader
# test_dataset = train_network.test_dataset
# # print(data.shape)
# nn.evaluate_model(linear_model, test_dataset)
