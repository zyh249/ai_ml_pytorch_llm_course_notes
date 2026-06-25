"""全连接神经网络参数量计算。"""
def linear_params(in_features, out_features, bias=True):
    return in_features * out_features + (out_features if bias else 0)

layers = [20, 128, 256, 4]
total = 0
for i, (din, dout) in enumerate(zip(layers, layers[1:]), 1):
    count = linear_params(din, dout)
    total += count
    print(f"layer{i}: ({din}+1)*{dout} = {count:,}")
print("total parameters:", f"{total:,}")
