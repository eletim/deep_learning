import argparse, os, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# ------------------------------
# Config
# ------------------------------
DATA_DIR = "./data"
OUT_DIR = Path("runs")
Z_DIM = 16
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# VAE（Conv Encoder/Decoder）
# ------------------------------
class Encoder(nn.Module):
    # 入力: (N,1,28,28) → 出力: μ, logσ^2 (各 Z_DIM)
    def __init__(self, z_dim=Z_DIM):
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
        self.fc_mu = nn.Linear(64*7*7, z_dim)
        self.fc_logvar = nn.Linear(64*7*7, z_dim)

    def forward(self, x):
        h = self.conv(x)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    # 入力: z (N,Z_DIM) → 出力: (N,1,28,28)
    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.fc = nn.Linear(z_dim, 64*7*7)
        # 7x7 → 14x14 → 28x28 にアップサンプル
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # 28x28
            nn.Sigmoid(),  # 0-1 に収める（BCE用）
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 64, 7, 7)
        xhat = self.deconv(h)
        return xhat


class VAE(nn.Module):
    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.enc = Encoder(z_dim)
        self.dec = Decoder(z_dim)

    def reparameterize(self, mu, logvar):
        # z = mu + eps * sigma
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.dec(z)
        return xhat, mu, logvar


# ------------------------------
# Loss（再構成BCE + KLD）
# ------------------------------
def vae_loss(xhat, x, mu, logvar):
    # 再構成誤差：画素ごとのBCEを総和
    bce = nn.functional.binary_cross_entropy(xhat, x, reduction="sum")
    # KLD：q(z|x) と N(0, I) のKLダイバージェンス
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld), bce, kld


# ------------------------------
# Data
# ------------------------------
def get_loaders():
    tfm = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
    ])
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
def train(vae, train_loader, test_loader, out_dir):
    vae = vae.to(DEVICE)
    opt = optim.Adam(vae.parameters(), lr=LR)
    best_total = math.inf
    out_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, EPOCHS+1):
        vae.train()
        total = 0.0; total_bce = 0.0; total_kld = 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            xhat, mu, logvar = vae(x)
            loss, bce, kld = vae_loss(xhat, x, mu, logvar)
            loss.backward()
            opt.step()
            total += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

        # 検証（テストセットでの平均損失）
        vae.eval()
        with torch.no_grad():
            val_total = 0.0
            for x, _ in test_loader:
                x = x.to(DEVICE)
                xhat, mu, logvar = vae(x)
                loss, _, _ = vae_loss(xhat, x, mu, logvar)
                val_total += loss.item()

        n_train = len(train_loader.dataset)
        n_test  = len(test_loader.dataset)
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"train_loss/px: {(total/n_train):.4f} "
              f"(bce {(total_bce/n_train):.4f} + kld {(total_kld/n_train):.4f}) | "
              f"val_loss/px: {(val_total/n_test):.4f} "
              f"device={DEVICE}")

        # ベストモデル保存（検証損失が最小）
        if val_total < best_total:
            best_total = val_total
            torch.save({"model": vae.state_dict(), "z_dim": Z_DIM}, out_dir/"vae_best.pt")

        # 各epochでサンプルを保存
        with torch.no_grad():
            z = torch.randn(64, Z_DIM, device=DEVICE)
            samples = vae.dec(z).cpu()
            utils.save_image(samples, out_dir/f"vae_sample_epoch{epoch}.png", nrow=8, padding=2)

    print(f"[INFO] Best model saved to {out_dir/'vae_best.pt'}")


def load_weights(vae, path):
    ckpt = torch.load(path, map_location=DEVICE)
    vae.load_state_dict(ckpt["model"])
    return vae


# ------------------------------
# Generation helpers
# ------------------------------
@torch.no_grad()
def save_random_samples(vae, out_dir, n=64, nrow=8):
    vae.eval()
    z = torch.randn(n, Z_DIM, device=DEVICE)
    imgs = vae.dec(z).cpu()
    utils.save_image(imgs, out_dir/"vae_sample.png", nrow=nrow)
    print(f"[SAVE] {out_dir/'vae_sample.png'}")


@torch.no_grad()
def save_latent_traversal(vae, out_dir, steps=8, span=3.0):
    """
    潜在ベクトルの各次元を -span～+span で等間隔に動かし、
    1次元ずつ変化させたときの生成画像を並べて保存。
    """
    vae.eval()
    base = torch.zeros(1, Z_DIM, device=DEVICE)
    rows = []
    for d in range(Z_DIM):
        zs = []
        for t in torch.linspace(-span, +span, steps, device=DEVICE):
            z = base.clone()
            z[0, d] = t
            zs.append(vae.dec(z).cpu())
        row = torch.cat(zs, dim=0)
        rows.append(row)
    grid = torch.cat(rows, dim=0)  # (Z_DIM*steps, 1, 28, 28)
    utils.save_image(grid, out_dir/"vae_traverse.png", nrow=steps)
    print(f"[SAVE] {out_dir/'vae_traverse.png'}")

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
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    train_loader, test_loader = get_loaders()

    vae = VAE(Z_DIM).to(DEVICE)

    if args.eval_only:
        if not args.weights or not os.path.exists(args.weights):
            raise FileNotFoundError("--eval-only には --weights を指定してください")
        load_weights(vae, args.weights)
        save_random_samples(vae, OUT_DIR)
        save_latent_traversal(vae, OUT_DIR)
        return

    # 学習
    train(vae, train_loader, test_loader, OUT_DIR)

    # ベスト重みで最終生成
    load_weights(vae, OUT_DIR/"vae_best.pt")
    save_random_samples(vae, OUT_DIR)
    save_latent_traversal(vae, OUT_DIR)

    Z, Y = collect_latents(vae, test_loader, DEVICE, is_conditional=False)
    plot_latents(Z, Y, out_path="runs/vae_latent_tsne.png", method="tsne")
    plot_latents(Z, Y, out_path="runs/vae_latent_umap.png", method="umap")


if __name__ == "__main__":
    main()
