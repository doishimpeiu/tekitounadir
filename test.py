from PIL import Image
import numpy as np

### 学習済みパラメータの読み込み
W1 = np.load('fc1_weight.npy').T
B1 = np.load('fc1_bias.npy')
W2 = np.load('fc2_weight.npy').T
B2 = np.load('fc2_bias.npy')

### 対象画像の読み込み
with Image.open('0.bmp') as im:
    im = im.convert('L')              # グレー画像として取り出す
    im = im.resize((28,28))           # 28x28 に画像をリサイズ
    im = np.asarray(im)               # ndarray として取り出す
A1 = (im.reshape(-1, 28*28)-128.0)/128.0            # (28, 28) の ndarray を (1, 784) に reshape #値域を0～255から-1.0～+1.0にスケーリング

### 推論処理
Y1 = np.dot(A1,W1) + B1
print("shape")
print(Y1.shape)
A2 = np.maximum(Y1, np.zeros(Y1.shape))
print("shape")
print(A2.shape)
Y2 = np.dot(A2,W2) + B2

### 推論結果の出力
print(Y2)
print(f"Result = {np.argmax(Y2)}")