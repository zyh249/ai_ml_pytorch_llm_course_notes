"""在二维二次函数上比较 SGD 与 Momentum。"""
def grad(x,y): return 0.25*x, 4.0*y
def loss(x,y): return 0.125*x*x + 2.0*y*y
for method in ["sgd","momentum"]:
    x,y,vx,vy=5.0,4.0,0.0,0.0
    for _ in range(30):
        gx,gy=grad(x,y)
        if method=="sgd": dx,dy=0.18*gx,0.18*gy
        else:
            vx=0.85*vx+0.15*gx; vy=0.85*vy+0.15*gy; dx,dy=0.32*vx,0.32*vy
        x,y=x-dx,y-dy
    print(method, (round(x,4),round(y,4)), round(loss(x,y),6))
