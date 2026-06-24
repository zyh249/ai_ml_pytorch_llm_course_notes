"""XGBoost 二阶近似与叶子最优权重的简化计算。"""
# 对一个叶节点，给定样本的一阶梯度 g_i 和二阶梯度 h_i：
# 最优叶子权重 w* = -sum(g) / (sum(h) + lambda)
# 叶子得分 gain = 0.5 * sum(g)^2 / (sum(h) + lambda)

gradients = [-0.8, -0.3, 0.2, -0.5]
hessians = [0.25, 0.21, 0.16, 0.24]
reg_lambda = 1.0
G, H = sum(gradients), sum(hessians)
weight = -G / (H + reg_lambda)
gain = 0.5 * G * G / (H + reg_lambda)
print('G =', round(G, 4), 'H =', round(H, 4))
print('optimal leaf weight =', round(weight, 4))
print('leaf gain =', round(gain, 4))
