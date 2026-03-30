import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CNN(nn.Module):
    """
    Improved CNN for MNIST-like (1x28x28) input.
    Architecture:
      - conv1(1->32, 3x3) -> bn1 -> relu -> pool  => 14x14
      - conv2(32->64, 3x3) -> bn2 -> relu -> pool  => 7x7
      - conv3(64->128, 3x3) -> bn3 -> relu -> pool => 3x3
      - fc1(128*3*3=1152 -> 256) -> relu -> dropout
      - fc2(256 -> 10)
      - log_softmax output
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def set_seed(seed=42):
    """Set RNG seeds for python, numpy and torch to improve reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train model for a single epoch over loader. Returns average loss."""
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / max(1, len(loader.dataset))


def evaluate(model, loader, device):
    """Evaluate model accuracy (fraction correct) on given loader."""
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return float(correct / total) if total > 0 else 0.0


def main(args):
    """Main entrypoint: trains CNN on MNIST and saves best model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    set_seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    num_workers = 0
    train_ds = datasets.MNIST(root='.', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root='.', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)

    model     = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc  = evaluate(model, test_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:02d}  Train loss: {loss:.4f}  Test Acc: {acc*100:.4f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "improved_digit_cnn.pth")

    print(f"Best Test Accuracy during run: {best_acc*100:.4f}%")
    print("Model saved to improved_digit_cnn.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Improved MNIST CNN training script")
    parser.add_argument('--epochs',     type=int,   default=15)
    parser.add_argument('--batch-size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()
    main(args)
