import argparse, os, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# ------------------------------
# Config
# ------------------------------
DATA_DIR = "./data"
OUT_DIR = Path("runs")
Z_DIM = 16          # 潜在次元
NUM_CLASSES = 10    # MNIST 0-9
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_hot(labels, num_classes=NUM_CLASSES):
    # labels: (N,) int64 -> (N, C) float
    y = torch.zeros(labels.size(0), num_classes, device=labels.device)
    y.scatter_(1, labels.view(-1,1), 1.0)
    return y


# ------------------------------
# CVAE（Encoder/Decoder ともにラベル条件を連結）
# ------------------------------
class Encoder(nn.Module):
    """
    入力: x (N,1,28,28), y (N,10 one-hot)
    処理: x -> conv -> flatten し、y を連結 → mu, logvar を出力
    """
    def __init__(self, z_dim=Z_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28
            nn.ReLU(True),
            nn.MaxPool2d(2),                 # 14x14
            nn.Conv2d(32, 64, 3, padding=1), # 14x14
            nn.ReLU(True),
            nn.MaxPool2d(2),                 # 7x7
        )
        self.flatten = nn.Flatten()
        in_fc = 64*7*7 + num_classes
        self.fc_mu = nn.Linear(in_fc, z_dim)
        self.fc_logvar = nn.Linear(in_fc, z_dim)

    def forward(self, x, y_onehot):
        h = self.conv(x)
        h = self.flatten(h)                   # (N, 64*7*7)
        h = torch.cat([h, y_onehot], dim=1)   # (N, 64*7*7 + C)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    入力: z (N,Z), y (N,10 one-hot)
    処理: [z; y] を全結合で 64*7*7 に変換 → 反転畳み込みで 28x28 に復元
    出力: xhat (N,1,28,28) in [0,1]
    """
    def __init__(self, z_dim=Z_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(z_dim + num_classes, 64*7*7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 28x28
            nn.Sigmoid(),  # 0-1（BCE用）
        )

    def forward(self, z, y_onehot):
        zy = torch.cat([z, y_onehot], dim=1)   # (N, Z+C)
        h = self.fc(zy)
        h = h.view(-1, 64, 7, 7)
        xhat = self.deconv(h)
        return xhat


class CVAE(nn.Module):
    def __init__(self, z_dim=Z_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc = Encoder(z_dim, num_classes)
        self.dec = Decoder(z_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, y_onehot):
        mu, logvar = self.enc(x, y_onehot)
        z = self.reparameterize(mu, logvar)
        xhat = self.dec(z, y_onehot)
        return xhat, mu, logvar


# ------------------------------
# Loss（再構成BCE + KLD）
# ------------------------------
def cvae_loss(xhat, x, mu, logvar):
    bce = nn.functional.binary_cross_entropy(xhat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld), bce, kld


# ------------------------------
# Data
# ------------------------------
def get_loaders():
    tfm = transforms.ToTensor()  # [0,1]
    train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader


# ------------------------------
# Train / Eval
# ------------------------------
def train_epoch(cvae, loader, opt):
    cvae.train()
    total=total_bce=total_kld=0.0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y1 = one_hot(y)

        opt.zero_grad()
        xhat, mu, logvar = cvae(x, y1)
        loss, bce, kld = cvae_loss(xhat, x, mu, logvar)
        loss.backward()
        opt.step()

        total += loss.item()
        total_bce += bce.item()
        total_kld += kld.item()
    n = len(loader.dataset)
    return total/n, total_bce/n, total_kld/n


@torch.no_grad()
def eval_epoch(cvae, loader):
    cvae.eval()
    total=0.0
    for x, y in loader:
        x = x.to(DEVICE)
        y1 = one_hot(y.to(DEVICE))
        xhat, mu, logvar = cvae(x, y1)
        loss, _, _ = cvae_loss(xhat, x, mu, logvar)
        total += loss.item()
    n = len(loader.dataset)
    return total/n


def train(cvae, train_loader, test_loader, out_dir, epochs):
    out_dir.mkdir(parents=True, exist_ok=True)
    cvae = cvae.to(DEVICE)
    opt = optim.Adam(cvae.parameters(), lr=LR)
    best = math.inf

    for ep in range(1, epochs+1):
        tr_total, tr_bce, tr_kld = train_epoch(cvae, train_loader, opt)
        val_total = eval_epoch(cvae, test_loader)
        print(f"Epoch {ep}/{epochs} | train: {tr_total:.4f} (bce {tr_bce:.4f} + kld {tr_kld:.4f}) | val: {val_total:.4f} | device={DEVICE}")

        # ベスト保存
        if val_total < best:
            best = val_total
            torch.save({"model": cvae.state_dict(), "z_dim": Z_DIM}, out_dir/"cvae_best.pt")

        # 各エポックで全クラスのサンプル保存
        save_all_classes_grid(cvae, out_dir, fname=f"cvae_ep{ep:02d}.png")

    print(f"[INFO] Best model saved to {out_dir/'cvae_best.pt'}")


def load_weights(cvae, path):
    ckpt = torch.load(path, map_location=DEVICE)
    cvae.load_state_dict(ckpt["model"])
    return cvae


# ------------------------------
# Generation helpers
# ------------------------------
@torch.no_grad()
def save_class_samples(cvae, out_dir, digit=7, n=64, nrow=8, fname="cvae_digit.png"):
    """特定の数字だけを生成（ラベル条件つき）"""
    cvae.eval()
    z = torch.randn(n, Z_DIM, device=DEVICE)
    y = torch.full((n,), int(digit), device=DEVICE, dtype=torch.long)
    y1 = one_hot(y)  # (n,10)
    imgs = cvae.dec(z, y1).cpu()
    utils.save_image(imgs, out_dir/fname, nrow=nrow)
    print(f"[SAVE] {out_dir/fname}  (class={digit})")


@torch.no_grad()
def save_all_classes_grid(cvae, out_dir, n_per_class=8, fname="cvae_all.png"):
    """
    0〜9 を各行に並べた画像を保存。
    各行 n_per_class 枚、合計 10*n_per_class 枚
    """
    cvae.eval()
    rows = []
    for d in range(NUM_CLASSES):
        z = torch.randn(n_per_class, Z_DIM, device=DEVICE)
        y = torch.full((n_per_class,), d, device=DEVICE, dtype=torch.long)
        y1 = one_hot(y)
        imgs = cvae.dec(z, y1).cpu()  # (n,1,28,28)
        rows.append(imgs)
    grid = torch.cat(rows, dim=0)  # (10*n,1,28,28)
    utils.save_image(grid, out_dir/fname, nrow=n_per_class)
    print(f"[SAVE] {out_dir/fname}  (rows=0..9)")


# ---- 潜在ベクトルを集める関数 ----
@torch.no_grad()
def collect_latents(model, loader, device, is_conditional=False, num_classes=10):
    model.eval()
    Z_list, Y_list = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if is_conditional:
            # CVAE の場合（既存の one_hot ヘルパを利用）
            y_onehot = one_hot(y, num_classes)
            mu, logvar = model.enc(x, y_onehot)
        else:
            # VAE の場合
            mu, logvar = model.enc(x)
        Z_list.append(mu.cpu())
        Y_list.append(y.cpu())  # ★ CPU に戻してから蓄積
    Z = torch.cat(Z_list, dim=0).numpy()
    Y = torch.cat(Y_list, dim=0).numpy()
    return Z, Y

# ---- 可視化関数 ----
def plot_latents(Z, Y, out_path="latent_tsne.png", method="tsne"):
    if method == "tsne":
        tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
        emb = tsne.fit_transform(Z)
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(Z)
    else:
        raise ValueError("method must be 'tsne' or 'umap'")

    plt.figure(figsize=(6,6))
    scatter = plt.scatter(emb[:,0], emb[:,1], c=Y, s=5, cmap="tab10")
    plt.colorbar(scatter, ticks=range(10))
    plt.title(f"Latent space ({method.upper()})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVE] {out_path}")


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--class", dest="klass", type=int, default=-1, help="生成したい数字(0-9)")
    parser.add_argument("--all-classes", action="store_true", help="0-9を行ごとに生成")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    # 🟢 修正ポイント: globalをやめてローカル変数にする
    epochs = args.epochs

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = get_loaders()
    cvae = CVAE(Z_DIM, NUM_CLASSES).to(DEVICE)

    if args.eval_only:
        if not args.weights or not os.path.exists(args.weights):
            raise FileNotFoundError("--eval-only には --weights を指定してください")
        load_weights(cvae, args.weights)
        if args.all_classes:
            save_all_classes_grid(cvae, OUT_DIR, n_per_class=8, fname="cvae_all_eval.png")
        if args.klass >= 0:
            save_class_samples(cvae, OUT_DIR, digit=args.klass, n=64, nrow=8, fname=f"cvae_digit{args.klass}.png")
        if not args.all_classes and args.klass < 0:
            save_class_samples(cvae, OUT_DIR, digit=7, n=64, nrow=8, fname="cvae_digit7.png")
        return

    # 学習
    train(cvae, train_loader, test_loader, OUT_DIR, epochs)

    # ベスト重みで生成
    load_weights(cvae, OUT_DIR/"cvae_best.pt")
    save_all_classes_grid(cvae, OUT_DIR, n_per_class=8, fname="cvae_all_final.png")
    save_class_samples(cvae, OUT_DIR, digit=7, n=64, nrow=8, fname="cvae_digit7_final.png")

    Z, Y = collect_latents(cvae, test_loader, DEVICE, is_conditional=True)
    plot_latents(Z, Y, out_path="runs/cvae_latent_tsne.png", method="tsne")
    plot_latents(Z, Y, out_path="runs/cvae_latent_umap.png", method="umap")


if __name__ == "__main__":
    main()
