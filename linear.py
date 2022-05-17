def linear(x, weight, bias, in_features, out_features, y):
  for i in range(out_features):  #range()でいいか
    sum = 0.0
    for j in range(in_features):  #range()でいいか
      sum += x[j] * weight[i * in_features + j]
    y[i] = sum + bias[i]