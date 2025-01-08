import matplotlib.pyplot as plt
import numpy as np
import neural_network as nn


def test_mnist_data():

    train_dataset, test_dataset = nn.load_mnist()

    print(len(train_dataset))
    print(len(test_dataset))

    for i in [0, 5]:
        im, label = train_dataset[i]

        plt.imshow(im.numpy()[0])
        plt.title(label)
        plt.show()

