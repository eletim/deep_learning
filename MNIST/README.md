# 必要パッケージ
pip install torch torchvision matplotlib

# 実行

## 学習＆評価＆保存
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


## 学習済みモデルで推論だけ行いたい場合
python3 mnist_cnn.py --eval-only --weights runs/best.pt

```log
[EVAL] test_acc=99.28% | test_loss=0.0211
```
