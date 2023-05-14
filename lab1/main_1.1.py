import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import Subset
from torchvision.datasets import MNIST

from lab1.aux_functions import create_summary_writer
from models import MLP
from trainer import Trainer

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    batch_size = 2048
    lr = 0.001
    epochs = 5
    sizes = [128, 64, 10]  # layers size for the MLP

    writer = create_summary_writer(lr, batch_size, epochs, mode="mlp", sizes=sizes)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = MNIST('./data', train=False, download=True, transform=transform)
    # shuffle false per la riproducibilità
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    image, label = train_dataset[0]
    flatten_dim = image.shape[1] * image.shape[2]

    model = MLP(sizes, flatten_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, device, writer)

    for epoch in range(1, epochs + 1):
        trainer.train(train_loader, optimizer, criterion, epoch)
        trainer.test(test_loader, criterion, epoch)
