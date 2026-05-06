"""
practice_05_cnn_training.py - 2D CNN 실제 훈련 예시

sklearn digits 8x8 이미지 데이터를 2D CNN으로 분류합니다.
forward 내부 shape와 중간 결과는 디버거로 직접 확인하세요.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(42)
np.random.seed(42)


class DigitCNN(nn.Module):
    """입력: (B, 1, 8, 8), 출력: (B, 10)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 2 * 2, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # 여기에 break-point를 걸고 x, h1, h2, h3, h4, out의 shape를 확인하세요.
        h1 = F.relu(self.conv1(x)) 
        h2 = self.pool(h1)           
        h3 = F.relu(self.conv2(h2))  
        h4 = self.pool(h3)           
        out = h4.flatten(1)       
        out = F.relu(self.fc1(out))
        out = self.fc2(out)          
        return out


def load_digit_data(batch_size=64):
    digits = load_digits()
    X = digits.images.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    X = X[:, None, :, :]  # Conv2d 입력 형태: (N, C, H, W)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_data = (torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_data = (torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_data, test_data


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def train(model, train_loader, val_data, epochs=30, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    X_val, y_val = val_data

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            # 실제 훈련 forward/backward 흐름입니다.
            # model(X_batch) 안으로 Step Into(F11) 하면 DigitCNN.forward()를 확인할 수 있습니다.
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss /= total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, X_val, y_val)

        print(
            f"epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )


def main():
    train_loader, val_data, test_data = load_digit_data(batch_size=64)

    model = DigitCNN()
    print(model)

    train(model, train_loader, val_data, epochs=30, lr=1e-3)

    X_test, y_test = test_data
    test_loss, test_acc = evaluate(model, X_test, y_test)
    print(f"\ntest_loss={test_loss:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
