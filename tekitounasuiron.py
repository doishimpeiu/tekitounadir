import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import numpy as np


# 1. ネットワークモデルの定義
class Net(nn.Module):
    def __init__(self, num_output_classes=10):
        super(Net, self).__init__()

        # 入力は28x28 のグレースケール画像 (チャネル数=1)
        # self.state_dict()['conv1.weight'][0] = torch.tensor([conv1の新しいパラメータ])
        # print(self.state_dict()['conv1.weight'])
        # 出力が8チャネルとなるような畳み込みを行う
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)

        # 活性化関数はReLU
        self.relu1 = nn.ReLU(inplace=True)

        # 画像を28x28から14x14に縮小する
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4ch -> 8ch, 14x14 -> 7x7
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全結合層
        # 8chの7x7画像を1つのベクトルとみなし、要素数32のベクトルまで縮小
        self.fc1 = nn.Linear(8 * 7 * 7, 32)
        self.relu3 = nn.ReLU(inplace=True)

        # 全結合層その2
        # 出力クラス数まで縮小
        self.fc2 = nn.Linear(32, num_output_classes)

    def forward(self, x):
        # with open('file1_0.txt', 'w') as f:
        #     for i in x[0]:
        #       print(i, file=f)
        # 1層目の畳み込み
        # 活性化関数 (activation) はReLU
        W1 = np.load('conv1_weight.npy')
        # print(W1)
        W1 = torch.from_numpy(W1.astype(np.float32)).clone()
        self.conv1.weight = nn.Parameter(W1)
        B1 = np.load('conv1_bias.npy')
        B1 = torch.from_numpy(B1.astype(np.float32)).clone()
        self.conv1.bias = nn.Parameter(B1)
        x = self.conv1(x)
        # print(f"ここのはず:{x.shape}")
        # with open('file1_1.txt', 'w') as f:
        #     for i in x[0]:
        #       print(i, file=f)
        x = self.relu1(x)
        # with open('file1_2.txt', 'w') as f:
        #     for i in x[0]:
        #       print(i, file=f)

        # 縮小
        x = self.pool1(x)
        # print(f"x.shape:{x.shape}")
        # with open('file1_3.txt', 'w') as f:
        #     for i in x[0]:
        #       print(i, file=f)

        # print(f"self.conv2.weight.shape:{self.conv2.weight.shape}")
        W2 = np.load('conv2_weight.npy')
        W2 = torch.from_numpy(W2.astype(np.float32)).clone()
        self.conv2.weight = nn.Parameter(W2)
        B2 = np.load('conv2_bias.npy')
        B2 = torch.from_numpy(B2.astype(np.float32)).clone()
        self.conv2.bias = nn.Parameter(B2)
        # 2層目+縮小
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        #shape:16, 8, 7, 7

        # フォーマット変換 (Batch, Ch, Height, Width) -> (Batch, Ch)
        x = x.view(x.shape[0], -1)
        #shape:16, 392
        # print(x.shape)
        print(f"self.fc1.weight.shape:{self.fc1.weight.shape}")
        W3 = np.load('fc1_weight.npy')
        print(f"W3.shape:{W3.shape}")
        W3 = torch.from_numpy(W3.astype(np.float32)).clone()
        self.fc1.weight = nn.Parameter(W3)
        print(f"self.fc1.weight.shape:{self.fc1.weight.shape}")
        B3 = np.load('fc1_bias.npy')
        B3 = torch.from_numpy(B3.astype(np.float32)).clone()
        self.fc1.bias = nn.Parameter(B3)
        # 全結合層
        x = self.fc1(x)
        x = self.relu3(x)
        print(f"self.fc2.weight.shape:{self.fc2.weight.shape}")
        W4 = np.load('fc2_weight.npy')
        print(f"W4.shape:{W4.shape}")
        W4 = torch.from_numpy(W4.astype(np.float32)).clone()
        self.fc2.weight = nn.Parameter(W4)
        print(f"self.fc2.weight.shape:{self.fc2.weight.shape}")
        B4 = np.load('fc2_bias.npy')
        B4 = torch.from_numpy(B4.astype(np.float32)).clone()
        self.fc2.bias = nn.Parameter(B4)
        x = self.fc2(x)

        return x


net = Net()


# print(f"W1:{W1}")
### 対象画像の読み込み
with Image.open('0.bmp') as im:
    im = im.convert('L')              # グレー画像として取り出す
    im = im.resize((28,28))           # 28x28 に画像をリサイズ
    im = np.asarray(im)               # ndarray として取り出す
# A1 = (im.reshape(-1, 28*28)-128.0)/128.0 
A1 = im.reshape(1, 1, 28, 28)

# 真っ黒の画像を作る
BLK = np.zeros(28*28)
BLK_1 = BLK.reshape(1, 1, 28, 28)

##テスト
# numpy to tensor
# x = torch.from_numpy(x.astype(np.float32)).clone()
input_im = torch.from_numpy(A1.astype(np.float32)).clone()
# input_im = torch.from_numpy(BLK_1.astype(np.float32)).clone()
# print(input_im)
input = input_im
# print(f"input:{input}")
print(f"input.shape:{input.shape}")
#### float32?
output = net(input)
print(output)
print(f"Result = {torch.argmax(output)}")