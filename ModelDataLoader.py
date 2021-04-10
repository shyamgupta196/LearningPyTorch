'''
Implementation of everything Learnt in past 5 days
'''

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.utils.data import DataLoader
SEED = 42

# re data
features, targets = make_classification(1000, 5, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, random_state=SEED)
# resizing the trains and tests
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
# make a custom dataset


class DataSet:
    def __init__(self, training_feature, training_target):
        super(DataSet, self).__init__()
        self.training_feature = training_feature
        self.training_target = training_target

    def __len__(self):
        return len(self.training_feature)

    def __getitem__(self, idx):
        return dict(
            FE_idx=self.training_feature[idx],
            TA_idx=self.training_target[idx])
# pass in dataloader
DataSet = DataSet(X_train, y_train)
train_loader = DataLoader(DataSet, batch_size=64)
# prepare a logistic regression model 
# since this is a classification problem
in_shape = X_train.shape[1]
out_shape = y_train.shape[1]
model = torch.nn.Linear(in_shape, out_shape)
criterion = torch.nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
# train the model on a training loop
EPOCHS = 100

for i in range(EPOCHS):
    for DATA in train_loader:
        FE = DATA['FE_idx']
        TA = DATA['TA_idx']
        # forwards
        preds = torch.sigmoid(model(FE))
        loss = criterion(preds, TA)
        # backwards for weight update
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
    if i % 10 == 0:
        print(f'loss: {loss}')
# accuracy check of the model
with torch.no_grad():
    test_predictions = torch.sigmoid(model(X_test)).round()
    acc = (test_predictions.eq(y_test).sum()/len(y_test))*100
    print('acc:', acc)
