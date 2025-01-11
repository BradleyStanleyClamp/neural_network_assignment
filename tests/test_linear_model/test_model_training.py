import neural_network as nn
import matplotlib.pyplot as plt


def test_training_class():
    epochs = 4
    learning_rate = 3e-3
    optimizer = "adam"
    batch_size = 64
    linear_model = nn.Linear_model()

    train_network = nn.Train_network(
        "mnist", learning_rate, optimizer, batch_size, epochs, linear_model
    )

    train_network.train_model(live_plot_bool=False)


# test_training_class()
