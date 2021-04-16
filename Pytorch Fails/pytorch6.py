#  BACKPROP MATH UNDERSTANDING
# DW/DX = DW/DY*DY/DX

# USING BASIC CHAIN RULE TO USE BACKPROP


import torch 

x = torch.tensor(1.0)
y=torch.tensor(2.0)

w= torch.tensor(1.0,requires_grad=True)

y_hat = w*x
loss = (y_hat-y)**2

loss.backward()

print(w.grad)

# GRAD_DESC WITH AUTOGRAD MANUALLY
import numpy as np
# f = w*x
# f = 2*x

x = np.array([1,2,3,4],dtype='float')
y = np.array([2,4,6,8],dtype='float')

w=0.0 #weights init

# model pred
def forward(x):
    return w*x

# loss = MSE
def loss(y,y_pred):
    return ((y_pred-y)**2).mean()
    
# then we cal gradients manually using loss
# MSE = 1/n * (w*x-y)**2
# d(mse)/dw = 1/n 2x(w*x - y)

def gradients(x,y,y_pred):
    return np.dot(2*x,y_pred-y).mean()

print(f'prediction before for f(5):{forward(5):.3f}')
    
n_iters = 20
lr = 1e-2
for epoch in range(n_iters):
    # forward pass
    y_pred = forward(x)
    # loss cal
    l = loss(y,y_pred)
    # gradients update with weights
    dw = gradients(x,y,y_pred)
    
    # update weights
    w -= lr*dw
    
    if epoch%3 ==0:
        print(f'after {epoch} epoch\'s weight\'s are {w:.6f} and  loss : {l:.5f}')
        
print(f'our final prediction\'s is f(5) == {forward(5)}')

# so we saw that with lr==1e-3 it learnt pretty slow but now with lr==1e-2 its almost got it correct 
# the same we will do now using autograd of pytorch 

print('using AUTOGRAD','-'*20)

import torch
# f = w*x
# f = 2*x

x = torch.tensor([1,2,3,4],dtype=torch.float32)
y = torch.tensor([2,4,6,8],dtype=torch.float32)

w=torch.tensor(0.0,requires_grad=True) #weights init req. grad will calc. gradients automatically

# model pred
def forward(x):
    return w*x

# loss = MSE
def loss(y,y_pred):
    return ((y_pred-y)**2).mean()
    
# then we cal gradients manually using loss
# MSE = 1/n * (w*x-y)**2
# d(mse)/dw = 1/n 2x(w*x - y)

# def gradients(x,y,y_pred):
#     return np.dot(2*x,y_pred-y).mean()

print(f'prediction before for f(5):{forward(5):.3f}')
    
n_iters = 30
lr = 1e-2
for epoch in range(n_iters):
    # forward pass
    y_pred = forward(x)
    # loss cal
    l = loss(y,y_pred)
    # gradients update with weights
    # dw = gradients(x,y,y_pred) # instead of this we just propagate it backwards using back. prop
    
    l.backward()
    
    
    # update weights
    with torch.no_grad():
        w -= lr*w.grad
        
    # we zero the gradients otherwise it adds up and messes things up
    
    w.grad.zero_()
    
    if epoch%3 ==0:
        print(f'after {epoch} epoch\'s weight\'s are {w:.6f} and  loss : {l:.5f}')
        
print(f'our final prediction\'s is f(5) == {forward(5)}')
