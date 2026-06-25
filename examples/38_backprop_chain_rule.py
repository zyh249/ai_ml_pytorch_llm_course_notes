"""标量网络的前向传播与链式法则反向传播。"""
import math
x, y = 1.5, 1.0
w1, b1, w2, b2 = 0.8, -0.2, 1.1, 0.1
z1 = w1*x+b1; h=max(0.0,z1); z2=w2*h+b2; p=1/(1+math.exp(-z2)); loss=-math.log(p)
dz2=p-y; dw2=dz2*h; dh=dz2*w2; dz1=dh if z1>0 else 0.0; dw1=dz1*x
print(f"forward: z1={z1:.4f}, h={h:.4f}, z2={z2:.4f}, p={p:.4f}, loss={loss:.4f}")
print(f"gradients: dw1={dw1:.4f}, dw2={dw2:.4f}")
