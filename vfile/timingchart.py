#固定
ksize = 3
height = 28
width = 28
in_channels = 1
out_channels = 4
f = open("timingchart.txt", "w")
for och in range(out_channels):
  for h in range(height):
    for w in range(width):
      sum = 0
      for ich in range(in_channels):
        for kh in range(ksize):
          for kw in range(ksize):
            ph = (h + kh - int(ksize /2)) 
            pw = (w + kw - int(ksize /2))
            pix_idx = ((ich * height + ph) * width + pw)      
            weight_idx = (((och * in_channels + ich) * ksize + kh) * ksize + kw)
            f.write(f"o : {och}, h : {h}, w : {w}, ich : {ich}, kh : {kh}, kw : {kw}, ph : {ph}, pw : {pw}, pix_idx : {pix_idx}, weight_idx : {weight_idx}\n")
      f.write(f"wが{w}の時\n")
      # input()
    f.write(f"hが{h}の時\n")
    # input()
  f.write(f"ochが{och}の時\n")
  # input()
    
f.close()