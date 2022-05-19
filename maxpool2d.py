import numpy as np
FLT_MAX = 100000000

def maxpool2d(x, width, height, channels, stride, y):
  x = x.reshape(4*28*28)
  # print(f"x.shape:{x.shape}")
  print(int(width/stride) * int(height/stride) * channels)
  y_tmp = np.empty(int(width/stride) * int(height/stride) * channels)
  print(f"y_tmp.shape{y_tmp.shape}")
  for ch in range(channels):  #range()でいいか
    for h in range(0, height, stride):  #range()でいいか w+=strideをどうするか→range(開始, 終了, インクリメント)
      for w in range(0, width, stride):
        maxval = -FLT_MAX #定義する
        
        for bh in range(stride):
          for bw in range(stride):
            # maxval = #std::max(maxval, x[(ch * height + h + bh) * width + w + bw]);
            maxval = np.maximum(maxval, x[(ch * height + h + bh) * width + w + bw])
            # print(maxval)

        #intに直さないといけない？
        #c++の除算は切り捨てのはず pythonも切り捨てなので大丈夫
        y_tmp[(ch * int(height / stride) + int(h / stride)) * int(width / stride) + int(w / stride)] = maxval
  print(f"y_tmp.shape:{y_tmp.shape}")
  y = y_tmp.reshape(channels, int(width/stride), int(height/stride))
  return y
