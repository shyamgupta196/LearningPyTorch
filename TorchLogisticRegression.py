'''
MAKING A LOGISTIC REGRESSION MODEL USING PYTORCH 
'''

import torch 
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import numpy as np
class vars:
    EPOCHS = 700
    RANDOM_STATE = 42
    LR = 1e-1
class LogReg(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(LogReg,self).__init__()
        self.linear = nn.Linear(input_shape,output_shape)
    def forward(self,x):
        prediction = torch.sigmoid(self.linear(x))
        return prediction
# PREP DATA
iris = load_breast_cancer()
inputs = iris.data
targets = iris.target

x_train,x_test,y_train,y_test = train_test_split(
    inputs,targets,
    shuffle = True,
    random_state = vars.RANDOM_STATE
    )
# data has a large variation in the values hence we SCALE THE DATA
# TRANSFORM VS FIT_TRANSFORM check out resources section of the post 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# from np array to tensors
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
y_train,y_test = y_train.view(y_train.shape[0],1),y_test.view(y_test.shape[0],1)

input_sample,input_shape = inputs.shape
print(inputs.shape)
output_shape = 1
# TRAIN THE MODEL FOR MAKING PREDICTIONS 
model = LogReg(input_shape,output_shape)
criterion = nn.L1Loss()
optimiser = torch.optim.SGD(model.parameters(),lr=vars.LR)
# LOOP
for i in range(vars.EPOCHS):
    # forward pass
    PREDICTIONS = model(x_train)
    # loss
    loss = criterion(y_train,PREDICTIONS)
    # backprop
    loss.backward()
    # optimise 
    optimiser.step()
    optimiser.zero_grad()
    if i%100 == 0:
        print(f'at {i} EPOCH loss = {loss:.3f}')

# evaluating model
with torch.no_grad(): 
    correct=0
    test_predictions = model(x_test).round()
    acc = (test_predictions.eq(y_test).sum()/len(y_test))*100
    print('acc:',acc)
#HOORAY WE GOT A 97% ACCURACY WITH THE MODEL    
