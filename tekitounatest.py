import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix


# 1. ネットワークモデルの定義
class Net(nn.Module):
    def __init__(self, num_output_classes=10):
        super(Net, self).__init__()

        # 入力は28x28 のグレースケール画像 (チャネル数=1)
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
        # 1層目の畳み込み
        # 活性化関数 (activation) はReLU
        x = self.conv1(x)
        # with open('file1.txt', 'w') as f:
        #     print(x, file=f)
        x = self.relu1(x)

        # 縮小
        x = self.pool1(x)

        # 2層目+縮小
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        #shape:16, 8, 7, 7

        # フォーマット変換 (Batch, Ch, Height, Width) -> (Batch, Ch)
        x = x.view(x.shape[0], -1)
        #shape:16, 392
        # print(x.shape)

        # 全結合層
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


net = Net()

# 2. データセットの読み出し法の定義
# MNIST の学習・テストデータの取得
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# データの読み出し方法の定義
# 1stepの学習・テストごとに16枚ずつ画像を読みだす
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

# ロス関数、最適化器の定義
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# 4. テスト
ans = []
pred = []
for i, data in enumerate(testloader, 0):
    # print(f"i:{i}", f"data:{data}")
    # input()
    inputs, labels = data
    print(f"inputs:{inputs.shape}") #inputs:torch.Size([16, 1, 28, 28])
    print(f"inputs:{inputs[0].shape}") #inputs:torch.Size([1, 28, 28])
    print(f"inputs:{inputs[0][0].shape}")
    # print(f"inputs:{inputs[1].shape}")
    # print(f"inputs:{inputs[2].shape}")
    # print(f"inputs:{inputs[3].shape}")
    # input()
    print(f"labels:{labels}")
    print(f"labels:{labels[0]}")
    input()
    
    outputs = net(inputs)

    ans += labels.tolist()
    pred += torch.argmax(outputs, 1).tolist()

print('accuracy:', accuracy_score(ans, pred))
print('confusion matrix:')
print(confusion_matrix(ans, pred))