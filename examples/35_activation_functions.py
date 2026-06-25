"""常见激活函数及导数。"""
import math

def sigmoid(x): return 1 / (1 + math.exp(-x))
def relu(x): return max(0.0, x)
def leaky_relu(x, negative_slope=0.1): return x if x >= 0 else negative_slope * x

for x in [-4, -1, 0, 1, 4]:
    s, t = sigmoid(x), math.tanh(x)
    print(x, "sigmoid=", round(s,4), "sigmoid'=", round(s*(1-s),4),
          "tanh=", round(t,4), "relu=", relu(x), "leaky=", leaky_relu(x))
