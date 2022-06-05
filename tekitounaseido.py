from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix
import conv2d
import relu
import maxpool2d
import linear

def cnn(im):
    A1 = im
    # A1 = BLK_1
    # print(f"A1.shape:{A1.shape}")
    # print(A1)

    y = np.empty(4*28*28)
    # with open('file2_0.txt', 'w') as f:
    #     for i in A1:
    #       print(i, file=f)
    ### 推論処理
    # print(f"W1.shape:{W1.shape}")
    # print(W1[4]) W1:3*3行列が4枚
    # conv2d(x, weight, bias, width, height, in_channels, out_channels, ksize, y)
    Y1 = conv2d.conv2d(A1, W1_int8, B1_int8, 28, 28, 1, 4, 3, y)
    # print(f"Y1.shape:{Y1.shape}")
    # with open('file2_1.txt', 'w') as f:
    #     for i in Y1:
    #       print(i, file=f)
    # print(f"Y1:{Y1}")
    # print(f"Y1.shape:{Y1.shape}")

    #reluに渡す時にreshapeするか、渡す前にreshapeするか
    Y1 = Y1.reshape(4*28*28)
    # print(f"Y1.shape:{Y1.shape}")
    # print(len(Y1))
    Y2 = relu.relu(Y1, len(Y1), y)
    # print(f"Y2.shape:{Y2.shape}")
    # with open('file2_2.txt', 'w') as f:
    #     for i in Y2:
    #       print(i, file=f)
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
    # with open('file2_3.txt', 'w') as f:
    #     for i in Y2:
    #       print(i, file=f)
    # print(f"Y3.shape:{Y3.shape}")

    ##
    # print(f"W2.shape{W2.shape}")
    # print(f"B2.shape{B2.shape}")
    y = np.empty(8*14*14)
    Y4 = conv2d.conv2d(Y3, W2_int8, B2_int8, 14, 14, 4, 8, 3, y)
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
    Y7 = linear.linear(Y6, W3_int8, B3_int8, 8*7*7, 32, y)
    # print(f"Y7.shape{Y7.shape}") #(32,)
    Y8 = relu.relu(Y7, len(Y7), y)
    # print(f"Y8.shape{Y8.shape}") #(32,)
    y = np.empty(10)
    Y9 = linear.linear(Y8, W4_int8, B4_int8, 32, 10, y)
    return Y9

### 学習済みパラメータの読み込み
# W1 = np.load('conv1_weight.npy').T ### .Tをつけると、3,3, 1, 4が4, 1, 3, 3になる
# , dtype = np.int8
W1 = np.load('conv1_weight.npy')
#最大値を出力
print(最大値を出力)
print(np.max(W1))
input()
# print(W1)
# input()
W1_int = ((W1+128)*128)
# print(W1_int)
# input()
W1_int8 = np.int8(((W1+128)*128))
# print(W1_uint8)
# input()
B1 = np.load('conv1_bias.npy')
B1_int8 = np.int8(((B1+128)*128))
W2 = np.load('conv2_weight.npy')
W2_int8 = np.int8(((W2+128)*128))
B2 = np.load('conv2_bias.npy')
B2_int8 = np.int8(((B2+128)*128))
W3 = np.load('fc1_weight.npy')
W3_int8 = np.int8(((W3+128)*128))
B3 = np.load('fc1_bias.npy')
B3_int8 = np.int8(((B3+128)*128))
W4 = np.load('fc2_weight.npy')
W4_int8 = np.int8(((W4+128)*128))
B4 = np.load('fc2_bias.npy')
B4_int8 = np.int8(((B4+128)*128))

#データの読み出し
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# 4. テスト
ans = []
pred = []
print(len(testloader))
for i, data in enumerate(testloader, 0):
    print(i)
    inputs_tensor, labels = data
    # print(inputs_tensor)
    # print(inputs_tensor*255)
    inputs_tensor_int = torch.tensor(inputs_tensor*255, dtype=torch.uint8)
    inputs_np = inputs_tensor_int.numpy()
    # print(inputs_np)
    # input()
    outputs = cnn(inputs_np)
    outputs_tensor = torch.from_numpy(outputs.astype(np.float32)).clone()

    ans += labels.tolist()
    pred.append(torch.argmax(outputs_tensor))
    if i == 100:
        break
    # input()
print('accuracy:', accuracy_score(ans, pred))
print('confusion matrix:')
print(confusion_matrix(ans, pred))