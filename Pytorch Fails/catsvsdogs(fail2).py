'''
IN THIS WE WILL SEE A
VERY EASY IMPLEMENTATION OF CNN
USING A CIFAR-10 DATASET
LATER WE WILL USE MUCH COMPLEX DATASET
'''
import PIL.Image as image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt
from torchvision import transforms
import  torch.nn.functional as F
from skimage.io import imread
import os

# prepare data
convert = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((30, 30)), transforms.Grayscale
     ()])

LABELS = {'CAT':0,'DOG':1}
class DataSet:
    def __init__(self,label,data,transform=None):
        self.data = data
        self.label = label
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        try:
            img = os.listdir(self.data)
            img = imread(img[idx],as_gray=True)
            if self.transform:
                img = self.transform(img)
            return img,self.label
        except Exception as e:
            pass
# dataloader
dogs_data = DataSet(LABELS['CAT'],'PetImages/Dog',transform=convert)
Cat_data = DataSet(LABELS['DOG'],'PetImages/Cat',transform=convert)
cat_img,cat_label = DataLoader(Cat_data,batch_size=32,shuffle=True)
print(cat_img)

dog_img,dog_label = DataLoader(dogs_data,batch_size=32,shuffle=True)

# make net
class Net(nn.Module):
    def __init__(self):
        super(self, Net).__init__()
        self.conv1 = nn.Conv2d(30, 64, 5, padding=1)
        self.conv2 = nn.Conv2d(64, 124, 3)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)))
        x = F.max_pool2d(F.relu(self.conv2(x)))
        x = F.softmax(nn.Linear(torch.flatten(x),512), dim=1)# dim=1 passes data in batches
        x = F.softmax(nn.Linear(512,64), dim=1)
        x = F.softmax(nn.Linear(64,1), dim=1)
        return x
# net pass
# vars
# train loop
'''
This is not working since i havent learn or seen till now how to load 
a image folder with different sub-categories folders 
😅😅 its 3 days i want to use this and  i do not know what to do 
i guess i have to search more about how do i make it possible
'''


