import torch

from tqdm import tqdm


class Trainer:
    def __init__(self, model, device, writer=None, conv_model=False):
        self.model = model
        self.device = device
        self.writer = writer
        self.conv_model = conv_model
        self.writer = writer

    def train(self, train_loader, optimizer, criterion, epoch):
        self.model.train()
        train_loss = 0
        batch_idx = 0
        for data, target in tqdm(train_loader, desc=f'Training epoch {epoch}', leave=True):
            batch_idx += 1
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            if not self.conv_model:
                output = self.model(data.view(data.size(0), -1))
            else:
                output = self.model(data)

            loss = criterion(output, target)
            batch_loss = loss.item()
            if self.writer is not None:
                self.writer.add_scalar('train/Batch-Loss', batch_loss, (epoch-1) * len(train_loader) + batch_idx)
            train_loss += batch_loss

            loss.backward()
            optimizer.step()

        train_loss /= batch_idx
        if self.writer is not None:
            self.writer.add_scalar('train/Loss', train_loss, epoch)

    @torch.no_grad()
    def test(self, test_loader, criterion, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        batch_idx = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f'Testing epoch {epoch}', leave=True):
                batch_idx += 1
                data, target = data.to(self.device), target.to(self.device)
                if not self.conv_model:
                    output = self.model(data.view(data.size(0), -1))
                else:
                   output = self.model(data)

                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= batch_idx
            accuracy = 100. * correct / len(test_loader.dataset)

            if self.writer is not None:
                self.writer.add_scalar('test/Loss', test_loss, epoch)
                self.writer.add_scalar('test/Accuracy', accuracy, epoch)

            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(test_loader.dataset), accuracy))


