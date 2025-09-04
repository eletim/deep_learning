# CNN

## 必要パッケージ
pip install torch torchvision matplotlib

## 実行

### 学習＆評価＆保存
python3 mnist_cnn.py

```log
100%|████████████████████████████████████████| 9.91M/9.91M [00:01<00:00, 7.11MB/s]
100%|█████████████████████████████████████████| 28.9k/28.9k [00:00<00:00, 170kB/s]
100%|████████████████████████████████████████| 1.65M/1.65M [00:01<00:00, 1.62MB/s]
100%|████████████████████████████████████████| 4.54k/4.54k [00:00<00:00, 6.69MB/s]
[INFO] device = cuda
Epoch 1/5 | train_loss: 0.2327 | test_loss: 0.0575 | test_acc: 98.21% | 3.1s
Epoch 2/5 | train_loss: 0.0736 | test_loss: 0.0415 | test_acc: 98.68% | 3.1s
Epoch 3/5 | train_loss: 0.0555 | test_loss: 0.0268 | test_acc: 99.10% | 3.1s
Epoch 4/5 | train_loss: 0.0453 | test_loss: 0.0364 | test_acc: 98.88% | 3.1s
Epoch 5/5 | train_loss: 0.0394 | test_loss: 0.0211 | test_acc: 99.28% | 3.1s
[INFO] Best test_acc = 99.28% | saved to runs/best.pt
[FINAL] test_acc=99.28% | test_loss=0.0211
```

### 学習済みモデルで推論だけ行いたい場合
python3 mnist_cnn.py --eval-only --weights runs/best.pt

```log
[EVAL] test_acc=99.28% | test_loss=0.0211
```

# VAE

## 必要パッケージ
pip install torch torchvision matplotlib

## 実行

### 学習＆評価＆保存
python3 mnist_vae.py

```log
Epoch 1/10 | train_loss/px: 151.3597 (bce 130.0605 + kld 21.2992) | val_loss/px: 118.9019 device=cuda
Epoch 2/10 | train_loss/px: 115.5656 (bce 91.5972 + kld 23.9684) | val_loss/px: 111.2577 device=cuda
Epoch 3/10 | train_loss/px: 110.7057 (bce 86.6905 + kld 24.0151) | val_loss/px: 108.4419 device=cuda
Epoch 4/10 | train_loss/px: 108.3713 (bce 84.4025 + kld 23.9687) | val_loss/px: 106.7671 device=cuda
Epoch 5/10 | train_loss/px: 106.9473 (bce 82.9599 + kld 23.9874) | val_loss/px: 105.6542 device=cuda
Epoch 6/10 | train_loss/px: 106.0014 (bce 81.9342 + kld 24.0672) | val_loss/px: 104.8676 device=cuda
Epoch 7/10 | train_loss/px: 105.2032 (bce 81.1242 + kld 24.0790) | val_loss/px: 104.0618 device=cuda
Epoch 8/10 | train_loss/px: 104.5622 (bce 80.4684 + kld 24.0938) | val_loss/px: 103.8850 device=cuda
Epoch 9/10 | train_loss/px: 103.9587 (bce 79.8895 + kld 24.0692) | val_loss/px: 103.0940 device=cuda
Epoch 10/10 | train_loss/px: 103.4705 (bce 79.4076 + kld 24.0628) | val_loss/px: 102.6264 device=cuda
[INFO] Best model saved to runs/vae_best.pt
[SAVE] runs/vae_sample.png
[SAVE] runs/vae_traverse.png
```

### 学習済み重みで生成だけ
python mnist_vae.py --eval-only --weights runs/vae_best.pt

