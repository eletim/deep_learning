import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random


@dataclass
class Config:
    batch_size: int = 128
    epochs: int = 5
    lr: float = 1e-3
    num_workers: int = 2
    data_dir: str = "./data"
    out_dir: str = "runs"
    seed: int = 42
    augment: bool = True


class SimpleCNN(nn.Module):
    """
    入力: 1x28x28（モノクロ）
    構成: Conv(1→32) → ReLU → MaxPool → Conv(32→64) → ReLU → MaxPool → FC(64*7*7→128) → ReLU → FC(128→10)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 → 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 28x28 → 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 → 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 14x14 → 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 64*7*7
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(cfg: Config):
    # 画像をTensor化＆0-1正規化＋平均/分散での標準化
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    train_tfms = [transforms.ToTensor(), normalize]
    if cfg.augment:
        # ちょい回転＆平行移動（MNISTではこれで汎化が少し上がることが多い）
        train_tfms = [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            normalize,
        ]

    train_dataset = datasets.MNIST(
        root=cfg.data_dir, train=True, download=True, transform=transforms.Compose(train_tfms)
    )
    test_dataset = datasets.MNIST(
        root=cfg.data_dir, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize])
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total, loss_sum / total


def train(cfg: Config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    set_seed(cfg.seed)
    train_loader, test_loader = get_loaders(cfg)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
            count += labels.size(0)

        train_loss = running / count
        test_acc, test_loss = evaluate(model, test_loader, device)
        dt = time.time() - t0

        print(f"Epoch {epoch}/{cfg.epochs} "
              f"| train_loss: {train_loss:.4f} "
              f"| test_loss: {test_loss:.4f} "
              f"| test_acc: {test_acc*100:.2f}% "
              f"| {dt:.1f}s")

        # ベストモデルを保存
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model": model.state_dict()}, best_path)

    print(f"[INFO] Best test_acc = {best_acc*100:.2f}% | saved to {best_path}")
    return best_path


@torch.no_grad()
def predict_some(model, loader, device, n=8):
    """テストセットから数枚取り出して可視化（学習の手応え確認用）"""
    model.eval()
    images, labels = next(iter(loader))
    images = images[:n].to(device)
    labels = labels[:n].to(device)
    logits = model(images)
    preds = logits.argmax(dim=1)

    # 可視化
    images_cpu = images.cpu()
    plt.figure(figsize=(10, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images_cpu[i].squeeze(0), cmap="gray")
        title = f"pred:{preds[i].item()}\ntrue:{labels[i].item()}"
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-augment", action="store_true",
                        help="データ拡張を無効化（デフォルトは有効）")
    parser.add_argument("--eval-only", action="store_true",
                        help="学習せずに評価/推論のみ行う")
    parser.add_argument("--weights", type=str, default="",
                        help="評価/推論で使う重み .pt")
    args = parser.parse_args()

    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        augment=not args.no_augment,
    )

    # 出力ディレクトリ
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_loaders(cfg)
    model = SimpleCNN().to(device)

    if args.eval_only:
        if not args.weights or not os.path.exists(args.weights):
            raise FileNotFoundError("--eval-only には --weights を指定してください")
        model = load_weights(model, args.weights, device)
        acc, loss = evaluate(model, test_loader, device)
        print(f"[EVAL] test_acc={acc*100:.2f}% | test_loss={loss:.4f}")
        predict_some(model, test_loader, device, n=8)
        return

    # 学習
    best_path = train(cfg, args)

    # ベスト重みで最終評価＆可視化
    model = load_weights(model, best_path, device)
    acc, loss = evaluate(model, test_loader, device)
    print(f"[FINAL] test_acc={acc*100:.2f}% | test_loss={loss:.4f}")
    predict_some(model, test_loader, device, n=8)


if __name__ == "__main__":
    main()
