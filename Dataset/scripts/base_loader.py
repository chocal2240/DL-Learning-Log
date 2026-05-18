import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_dataloader(root_path, dataset_name, batch_size=32, train=True):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if dataset_name == "mnist":
        dataset = datasets.MNIST(root=root_path, train=train, download=True, transform=trans)
    elif dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root=root_path, train=train, download=True, transform=trans)
    else:
        raise NotImplementedError
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)
    return loader