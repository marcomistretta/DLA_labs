import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

from lab1.aux_functions import create_summary_writer
from models import CNN, ResCNN
from trainer import Trainer

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 1, 1024
    # 5, 512
    # 10, 256
    # 20, 128
    batch_size = 256
    lr = 0.004
    epochs = 30

    depth = 10
    residual = True
    writer = create_summary_writer(lr, batch_size, epochs, folder="deep_cnns_work_worst", mode="cnn", depth=depth, residual=residual)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10('./data', train=False, download=True, transform=transform)
    # shuffle false per la riproducibilità
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                               persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                              persistent_workers=True)

    if residual:
        model = ResCNN(depth=depth).to(device)
    else:
        model = CNN(depth=depth).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, device, writer, conv_model=True)
    try:
        for epoch in range(1, epochs + 1):
            trainer.train(train_loader, optimizer, criterion, epoch)
            trainer.test(test_loader, criterion, epoch)
    finally:
        # playsound.playsound("mixkit-correct-answer-tone-2870.wav")
        torch.save(model, writer.log_dir.split("\\")[-1] + ".pt")
