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
    y = np.empty(4*28*28)
    # conv2d(x, weight, bias, width, height, in_channels, out_channels, ksize, y)
    Y1 = conv2d.conv2d(A1, W1_int8, B1_int8, 28, 28, 1, 4, 3, y)
    Y1 = Y1.reshape(4*28*28)
    Y2 = relu.relu(Y1, len(Y1), y)
    Y2 = Y2.reshape(4, 28, 28)
    Y3 = maxpool2d.maxpool2d(Y2, 28, 28, 4, 2, y)
    y = np.empty(8*14*14)
    Y4 = conv2d.conv2d(Y3, W2_int8, B2_int8, 14, 14, 4, 8, 3, y)
    Y4 = Y4.reshape(8*14*14)
    Y5 = relu.relu(Y4, len(Y4), y)
    Y5 = Y5.reshape(8, 14, 14)
    # maxpool2d(x, width, height, channels, stride, y)
    Y6 = maxpool2d.maxpool2d(Y5, 14, 14, 8, 2, 2)
    y = np.empty(32)
    # linear(x, weight, bias, in_features, out_features, y)
    Y7 = linear.linear(Y6, W3_int8, B3_int8, 8*7*7, 32, y)
    Y8 = relu.relu(Y7, len(Y7), y)
    y = np.empty(10)
    Y9 = linear.linear(Y8, W4_int8, B4_int8, 32, 10, y)
    return Y9

### 学習済みパラメータの読み込み
W1 = np.load('conv1_weight.npy')
W1_int8 = np.int8(((W1*128)))
B1 = np.load('conv1_bias.npy')
B1_int8 = np.int8(B1*2**8)
W2 = np.load('conv2_weight.npy')
W2_int8 = np.int8(((W2*128)))
B2 = np.load('conv2_bias.npy')
B2_int8 = np.int8(((B2*128)))
W3 = np.load('fc1_weight.npy')
W3_int8 = np.int8(((W3*128)))
B3 = np.load('fc1_bias.npy')
B3_int8 = np.int8(((B3*128)))
W4 = np.load('fc2_weight.npy')
W4_int8 = np.int8(((W4*128)))
B4 = np.load('fc2_bias.npy')
B4_int8 = np.int8(((B4*128)))

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
    inputs_tensor_int = torch.tensor(inputs_tensor*255, dtype=torch.uint8)
    inputs_np = inputs_tensor_int.numpy()
    outputs = cnn(inputs_np)
    outputs_tensor = torch.from_numpy(outputs.astype(np.float32)).clone()

    ans += labels.tolist()
    pred.append(torch.argmax(outputs_tensor))
print('accuracy:', accuracy_score(ans, pred))
print('confusion matrix:')
print(confusion_matrix(ans, pred))