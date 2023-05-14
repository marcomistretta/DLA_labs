import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10

from lab1.models import CNN, ResCNN

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    batch_size = 30

    depth = 10
    cnn_writer = SummaryWriter("./model/cnn-" + str(depth))
    res_writer = SummaryWriter("./model/res-" + str(depth))

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                               persistent_workers=True)

    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 10
    # cnn_model = torch.load("./model/cnn-ep5-lr0.001-bs512-depth5.pt")
    # residual_cnn_model = torch.load("./model/resnet_cnn-ep5-lr0.001-bs512-depth5-residual.pt")
    cnn_model = CNN(depth=depth).to(device)
    residual_cnn_model = ResCNN(depth=depth).to(device)

    cnn_optimizer = torch.optim.Adam(lr=0.001, params=cnn_model.parameters())
    res_optimizer = torch.optim.Adam(lr=0.001, params=residual_cnn_model.parameters())
    count = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            count += 1
            images, labels = images.to(device), labels.to(device)

            cnn_outputs = cnn_model(images)
            cnn_loss = loss_fn(cnn_outputs, labels)

            residual_cnn_outputs = residual_cnn_model(images)
            residual_cnn_loss = loss_fn(residual_cnn_outputs, labels)

            cnn_model.zero_grad()
            cnn_loss.backward()
            cnn_grad_mag = torch.mean(torch.abs(cnn_model.classifier.weight.grad))

            residual_cnn_model.zero_grad()
            residual_cnn_loss.backward()
            residual_cnn_grad_mag = torch.mean(torch.abs(residual_cnn_model.classifier.weight.grad))

            print("Epoch [{}/{}], Step [{}/{}], CNN Grad Magnitude: {:.4f}, Residual CNN Grad Magnitude: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), cnn_grad_mag, residual_cnn_grad_mag))
            cnn_writer.add_scalar("Grad Magnitude", cnn_grad_mag, epoch * len(train_loader) + i)
            res_writer.add_scalar("Grad Magnitude", residual_cnn_grad_mag, epoch * len(train_loader) + i)

            cnn_optimizer.step()
            res_optimizer.step()
            if count == 150:
                break

        break
