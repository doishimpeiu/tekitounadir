def maxpool2d(x, width, height, channels, stride, y):
  for ch in range(channels):  #range()でいいか
    for h in range(height):  #range()でいいか w+=strideをどうするか
      for w in range(width):
        maxval = -FLT_MAX #定義する
        
        for bh in range(stride):
          for bw in range(stride):
            maxval = #std::max(maxval, x[(ch * height + h + bh) * width + w + bw]);

        y[(ch * (height / stride) + (h / stride)) * (width / stride) + w / stride] = maxval
        