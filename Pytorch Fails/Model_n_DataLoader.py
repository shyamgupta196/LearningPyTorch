'''
Making a Dataloader and trying to Apply everything learnt in past 4 days
'''
from sklearn.datasets import make_classification
import torch
import torch.nn as nn
import numpy as np

data = make_classification(
    n_samples=100, n_features=5, random_state=1)


class Dataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        features = self.data[0]
        targets = self.data[1]
        features = torch.from_numpy(features)
        targets = torch.from_numpy(targets.reshape(targets.shape[0],))
        features.requires_grad = True
        return features, targets


loader = torch.utils.data.DataLoader(
    Dataset(data),
    batch_size=64)
epochs = 10


class NET(nn.Module):
    def __init__(self, inputs, outputs):
        super(NET, self).__init__()
        self.linear = nn.Linear(inputs, outputs)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


input_shape = len(data[0])
output_shape = 2  # since 2 classes is default in the make classification

model = NET(input_shape, output_shape)
criterion = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=0.001)
for i in range(epochs):
    for (inputs, outputs) in loader:
        preds = model(inputs)
        loss = criterion(preds, outputs)
        loss.backward()
        optim.step()
        optim.zero_grad()

    if i % 2 == 0:
        print(f'loss:{loss:.4f}')
'''
I HAVE TO FIGURE OUT HOW TO BALANCE THE INPUT AND 
OUTPUT SHAPE TO BE GIVEN INTO THE MODEL
AND IMPROVE THE CODE STRUCTURE A BIT 
'''
