#https://www.acri.c.titech.ac.jp/wordpress/archives/5992をみてPythonに直したもの
# 入力
# x: 入力画像。shape=(in_channels, height, width)
# weight: 重み係数。shape=(out_channels, in_channels, ksize, ksize)
# bias: バイアス値。 shape=(out_channels)

# 出力
# y: 出力画像。shape=(out_channels, height, width)

# パラメータ:
# width: 入力/出力画像の幅
# height: 入力/出力画像の高さ
# in_channels: 入力画像のチャネル数
# out_channels: 出力画像のチャネル数
# ksize: カーネルサイズ

from PIL import Image
import numpy as np

def conv2d(x, weight, bias, width, height, in_channels, out_channels, ksize, y):
    for och in range(out_channels):
        print(f"och:{och}")
        for h in range(height):
            print(f"h:{h}")
            for w in range(width):
                print(f"w:{w}")
                sum = 0.0

                for ich in range(in_channels):
                    print(f"ich:{ich}")
                    for kh in range(ksize):
                        print(f"kh:{kh}")
                        for kw in range(ksize):
                            print(f"kw:{kw}")
                            ph = h + kh - (ksize/2)
                            print(f"ph:{ph}")
                            pw = w + kw - (ksize/2)
                            print(f"pw:{pw}")

                            #zero padding
                            if (ph < 0 or ph >= height or pw < 0 or pw >= width):
                                continue
                            
                            pix_idx = int((ich * height + ph) * width + pw)
                            print(f"pix_idx:{pix_idx}")
                            weight_idx = int(((och * in_channels + ich) * ksize + kh) * ksize + kw)
                            print(f"weight_idx:{weight_idx}")

                            sum += x[pix_idx] * weight[weight_idx]
                
                #add bias
                sum += bias[och]

                y[(och * weight + h) * width + w] = sum

def main():
    ### 学習済みパラメータの読み込み
    weight = np.load('fc1_weight.npy').T
    bias = np.load('fc1_bias.npy')
    in_channels = 1
    out_channels = 4
    ksize = 3
    width = 14
    height = 14
    x = []
    y = []
    conv2d(x, weight, bias, width, height, in_channels, out_channels, ksize, y)

if __name__ == '__main__':
    main()