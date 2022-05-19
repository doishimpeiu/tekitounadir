from PIL import Image
import numpy as np
import conv2d
import relu
import maxpool2d

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

y = np.empty(4*28*28)
### 推論処理
# print(f"W1.shape:{W1.shape}")
# print(W1[4]) W1:3*3行列が4枚
# conv2d(x, weight, bias, width, height, in_channels, out_channels, ksize, y)
Y1 = conv2d.conv2d(im, W1, B1, 28, 28, 1, 4, 3, y)
# print(f"Y1:{Y1}")
print(f"Y1.shape:{Y1.shape}")

#reluに渡す時にreshapeするか、渡す前にreshapeするか
Y1 = Y1.reshape(4*28*28)
print(f"Y1.shape:{Y1.shape}")
print(len(Y1))
Y2 = relu.relu(Y1, len(Y1), y)
print(Y2)
print(f"Y2.shape:{Y2.shape}")
# for i in Y2:
#     if i < 0:
#         input()
#     else:
#         print(i)

Y2 = Y2.reshape(4, 28, 28)
print(f"Y2.shape:{Y2.shape}")
Y3 = maxpool2d.maxpool2d(Y2, 28, 28, 4, 2, y)
print(f"Y3.shape:{Y3.shape}")

