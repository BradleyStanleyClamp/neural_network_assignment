from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_mnist():
    train_dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    test_dataset = datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    return train_dataset, test_dataset


def get_mnist_loaders(train_dataset, test_dataset, batch_size, shuffle=True):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )  # this automatically batches up examples, adding a batch dimension
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
