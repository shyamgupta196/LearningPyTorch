'''
Today We will try to implement a Conv Net Architecture
'''
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt 
from PIL import  ImageShow
# variables 
EPOCHS = 3
# prepare data
train = torchvision.datasets.FashionMNIST(
    root='../data', download=False, train=True)
test = torchvision.datasets.FashionMNIST(
    root='../data', download=False, train=False)
# make a custom loader
class DataSet:
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image,labels = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, labels
        
train_loader = torch.utils.data.DataLoader(DataSet(train,transform=transforms.ToTensor()),batch_size=6,shuffle=True)

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    
    for enum,(image,label) in enumerate(train_loader):

        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(image[enum][:])
plt.show()# Pass it in a custom net
class Net(nn.Module):
    def __init__(self,inputs,output,kernel_size):
        super(Net,self).__init__()
        self.Conv1 = nn.Conv2d(inputs,output,kernel_size)
        self.Conv2 = nn.Conv2d(output,124,10)
        self.Conv3 = nn.Conv2d(124,10,20)
        self.fc1 = nn.Linear()
        

# declare the model
# loss and optim
# train
# test the data and output accuracy


