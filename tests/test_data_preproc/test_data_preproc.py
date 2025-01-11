import matplotlib.pyplot as plt
import numpy as np
import neural_network as nn


def test_mnist_data():

    train_dataset, test_dataset = nn.load_mnist()

    print(len(train_dataset))
    print(len(test_dataset))

    for i in [0, 5]:
        im, label = train_dataset[i]

        # print(im.numpy()[0])
        # plt.imshow(im.numpy()[0])
        # plt.title(label)
        # plt.show()


# test_mnist_data()


def test_train_loader():
    train_dataset, test_dataset = nn.load_mnist()

    train_loader, test_loader = nn.get_mnist_loaders(train_dataset, test_dataset, 1)

    # im, label = train_dataset[0]
    # print(im.numpy()[0])
    # plt.imshow(im.numpy()[0])
    # plt.title(label)
    # plt.show()

    # print(train_dataset.data[0, :, :])
    # print(train_dataset.targets[0])

    # for batch_idx, (val) in enumerate(train_loader):
    # print(val)
    # print(im.numpy()[0])
