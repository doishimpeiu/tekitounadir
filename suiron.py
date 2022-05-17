from PIL import Image
import numpy as np
import conv2d

### 学習済みパラメータの読み込み
# W1 = np.load('conv1_weight.npy').T
W1 = np.load('conv1_weight.npy')
B1 = np.load('conv1_bias.npy')
W2 = np.load('conv2_weight.npy').T
B2 = np.load('conv2_bias.npy')

### 対象画像の読み込み
with Image.open('0.bmp') as im:
    im = im.convert('L')              # グレー画像として取り出す
    im = im.resize((28,28))           # 28x28 に画像をリサイズ
    im = np.asarray(im)               # ndarray として取り出す
# A1 = (im.reshape(-1, 28*28)-128.0)/128.0            # (28, 28) の ndarray を (1, 784) に reshape #値域を0～255から-1.0～+1.0にスケーリング

y = []
### 推論処理
print(f"W1.shape:{W1.shape}")
print(W1[3])
# conv2d(x, weight, bias, width, height, in_channels, out_channels, ksize, y)
Y1 = conv2d.conv2d(im, W1, B1, 28, 28, 1, 4, 3, y)
print(Y1)

