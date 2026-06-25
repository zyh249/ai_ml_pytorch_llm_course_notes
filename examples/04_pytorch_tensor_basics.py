"""PyTorch 张量创建、类型转换、NumPy 互转、形状操作。"""
import numpy as np
try:
    import torch
except ImportError as exc:
    raise SystemExit("请先安装 PyTorch：pip install torch") from exc

# 1. 按数据创建张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("x =", x, "shape=", x.shape, "dtype=", x.dtype)

# 2. 指定类型；推荐使用 dtype 参数而不是 torch.Tensor(size) 未初始化写法
x_float = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print("x_float dtype =", x_float.dtype)

# 3. 与 NumPy 共享内存：from_numpy / tensor.numpy()
arr = np.array([10, 20, 30], dtype=np.float32)
t = torch.from_numpy(arr)
t[0] = 99
print("arr after tensor change =", arr)  # [99. 20. 30.]

# 4. 不共享内存：torch.tensor(arr) 会复制数据
arr2 = np.array([1, 2, 3])
t2 = torch.tensor(arr2)
t2[0] = 100
print("arr2 =", arr2, "t2 =", t2)

# 5. 形状操作：transpose 后内存可能不连续，view 前要 contiguous
m = torch.arange(6).reshape(2, 3)
mt = m.transpose(0, 1)
print("mt.is_contiguous() =", mt.is_contiguous())
print("reshape works:", mt.reshape(2, 3))
print("view after contiguous:", mt.contiguous().view(2, 3))
