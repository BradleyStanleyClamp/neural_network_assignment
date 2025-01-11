import neural_network as nn
import matplotlib.pyplot as plt


def plot_epoch_loss(ax, epochs_list, loss_list):
    # Update the graph
    ax.clear()  # Clear previous content
    ax.plot(epochs_list, loss_list, label="Training Loss", color="blue", marker="o")
    ax.set_title("Loss vs Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()


# def test_training_basic():
#     epochs = 20
#     learning_rate = 3e-3
#     batch_size = 64
#     linear_model = nn.Linear_model()

#     train_dataset, test_dataset = nn.load_mnist()
#     train_loader, test_loader = nn.get_mnist_loaders(
#         train_dataset, test_dataset, batch_size=batch_size
#     )

#     epochs_list = []
#     loss_list = []
#     # plt.ion()  # Turn on interactive mode
#     # fig, ax = plt.subplots()

#     for epoch in range(0, epochs):
#         linear_model.train_model(epoch, train_loader, learning_rate)

#         epochs_list.append(epoch)
#         # loss_list.append(epoch_average_loss)

#         # plot_epoch_loss(ax, epochs_list, loss_list)

#     # plt.ioff()  # Turn off interactive mode
#     # plt.show()


def training_class():
    epochs = 20
    learning_rate = 3e-3
    optimizer = "adam"
    batch_size = 64
    linear_model = nn.Linear_model()

    train_network = nn.Train_network(
        "mnist", learning_rate, optimizer, batch_size, epochs, linear_model
    )

    # print(linear_model.fc3.W[0, 0])
    train_network.train_model(live_plot_bool=True)
    # print(linear_model.fc3.W[0, 0])
    # print(train_network.model.fc3.W[0, 0])


training_class()


def single_data_training_class():
    epochs = 100
    learning_rate = 3e-3
    batch_size = 5000
    optimizer = "adam"
    linear_model = nn.Linear_model()

    train_network = nn.Train_network(
        "mnist", learning_rate, optimizer, batch_size, epochs, linear_model
    )

    # print(linear_model.fc3.W[0, 0])
    train_network.train_model(live_plot_bool=True, test_on_data=True)
    # print(linear_model.fc3.W[0, 0])
    # print(train_network.model.fc3.W[0, 0])


# single_data_training_class()
