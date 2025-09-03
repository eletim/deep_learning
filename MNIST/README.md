# 必要パッケージ
pip install torch torchvision matplotlib

# 実行

## 学習＆評価＆保存
python3 mnist_cnn.py

## 学習済みモデルで推論だけ行いたい場合
python3 mnist_cnn.py --eval-only --weights runs/best.pt
