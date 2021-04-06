'''
MAKING LINEAR REGRESSION MODEL WITH PYTORCH 
'''
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# making random x,y using datasets from sklearn

x,y = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

# x,y are numpy arrays with dtype=='double'
# we have to convert it to float and then into tensor 

x,y = torch.from_numpy(x.astype(np.float32)),torch.from_numpy(y.astype(np.float32))

y = y.view(y.shape[0],1)
no_samples,no_features = x.shape
input_feautures = no_features
output_feautures = y.shape[1]

model = nn.Linear(input_feautures,output_feautures)
print(model)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(),lr=0.001)

n_iters = 1000
# train loop
for i in range(n_iters):
    # forward pass
    preds = model(x)
    # loss
    loss = criterion(preds,
        y)
    # loss update using backprop(backwards pass)
    loss.backward()
    optimizer.step()
    # nullifying grads
    optimizer.zero_grad()
    # print(loss)
    
    if i%4 == 0:
        w,b = model.parameters()
        print(f'after {i} epoch\'s we have loss : {loss.item():.4f}')
print(w)

preds = model(x).detach().numpy()
plt.plot(x,y,'ro')
plt.plot(x,preds,'b')
plt.show()
