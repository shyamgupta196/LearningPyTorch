'''learning basics from python engineer'''

import torch

x = torch.randn(3,requires_grad=True)
print(x)

y= x+2
print(y)
# vector jacobian product is applied 
# chain rule applies jacobian vector product
z = (y*y*2).mean()
z.backward() #dz/dw (formula for Back.prop)
print(x.grad)# since x has requires_grad==True
#  we can see these grads here

# TO PREVENT FROM TRACKING GRADIENTS

# x.requires_grad_(False)
# y= x.detach()
# with torch.no_grad():
#     print(x+2)

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    output = (weights*3).sum()
    output.backward()
    print(weights.grad)
    # we want the grads to be same not add up 
    # hence wedo
    weights.grad.zero_()
