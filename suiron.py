from PIL import Image
import numpy as np
import conv2d
import relu
import maxpool2d
import linear

### 学習済みパラメータの読み込み
# W1 = np.load('conv1_weight.npy').T ### .Tをつけると、3,3, 1, 4が4, 1, 3, 3になる
W1 = np.load('conv1_weight.npy').T
B1 = np.load('conv1_bias.npy')
W2 = np.load('conv2_weight.npy').T
B2 = np.load('conv2_bias.npy')
W3 = np.load('fc1_weight.npy').T
B3 = np.load('fc1_bias.npy')
W4 = np.load('fc2_weight.npy').T
B4 = np.load('fc2_bias.npy')

### 対象画像の読み込み
with Image.open('1.bmp') as im:
    im = im.convert('L')              # グレー画像として取り出す
    im = im.resize((28,28))           # 28x28 に画像をリサイズ
    im = np.asarray(im)               # ndarray として取り出す
# A1 = (im.reshape(-1, 28*28)-128.0)/128.0            # (28, 28) の ndarray を (1, 784) に reshape #値域を0～255から-1.0～+1.0にスケーリング
A1 = im
y = np.empty(4*28*28)
with open('file2_0.txt', 'w') as f:
    for i in A1:
      print(i, file=f)
### 推論処理
# print(f"W1.shape:{W1.shape}")
# print(W1[4]) W1:3*3行列が4枚
# conv2d(x, weight, bias, width, height, in_channels, out_channels, ksize, y)
Y1 = conv2d.conv2d(A1, W1, B1, 28, 28, 1, 4, 3, y)
print(f"Y1.shape:{Y1.shape}")
with open('file2_1.txt', 'w') as f:
    for i in Y1:
      print(i, file=f)
# print(f"Y1:{Y1}")
# print(f"Y1.shape:{Y1.shape}")

#reluに渡す時にreshapeするか、渡す前にreshapeするか
Y1 = Y1.reshape(4*28*28)
# print(f"Y1.shape:{Y1.shape}")
# print(len(Y1))
Y2 = relu.relu(Y1, len(Y1), y)
print(f"Y2.shape:{Y2.shape}")
with open('file2_2.txt', 'w') as f:
    for i in Y2:
      print(i, file=f)
# print(Y2)
# print(f"Y2.shape:{Y2.shape}")
# for i in Y2:
#     if i < 0:
#         input()
#     else:
#         print(i)

Y2 = Y2.reshape(4, 28, 28)
# print(f"Y2.shape:{Y2.shape}")
Y3 = maxpool2d.maxpool2d(Y2, 28, 28, 4, 2, y)
with open('file2_3.txt', 'w') as f:
    for i in Y2:
      print(i, file=f)
# print(f"Y3.shape:{Y3.shape}")

##
# print(f"W2.shape{W2.shape}")
# print(f"B2.shape{B2.shape}")
y = np.empty(8*14*14)
Y4 = conv2d.conv2d(Y3, W2, B2, 14, 14, 4, 8, 3, y)
# print(f"Y4.shape{Y4.shape}") #(8, 14, 14)
Y4 = Y4.reshape(8*14*14)
# print(f"Y4.shape{Y4.shape}") #(1568,)
# print(len(Y4))
Y5 = relu.relu(Y4, len(Y4), y)
# print(f"Y5.shape{Y5.shape}")
Y5 = Y5.reshape(8, 14, 14)
# maxpool2d(x, width, height, channels, stride, y)
Y6 = maxpool2d.maxpool2d(Y5, 14, 14, 8, 2, 2)
# print(f"Y6.shape{Y6.shape}") #(8, 7, 7)
#本来はここでのshapeが16, 8, 7, 7になっているはず？→16はバッチ数

##Reshape
# Y6 = Y6.view(Y6.shape[0], -1)

y = np.empty(32)
###reshape(2)とは
# linear(x, weight, bias, in_features, out_features, y)
Y7 = linear.linear(Y6, W3, B3, 8*7*7, 32, y)
# print(f"Y7.shape{Y7.shape}") #(32,)
Y8 = relu.relu(Y7, len(Y7), y)
# print(f"Y8.shape{Y8.shape}") #(32,)
y = np.empty(10)
Y9 = linear.linear(Y8, W4, B4, 32, 10, y)

### 推論結果の出力
print(f"Result = {np.argmax(Y9)}")