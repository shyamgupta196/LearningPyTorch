'''
In this TorchDaily we will see a 
simple convlution network using

Cats Vs Dogs Dataset    
'''
import PIL.Image as image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms,datasets
from tqdm import tqdm
import  torch.nn.functional as F
from skimage.io import imread
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# prepare data
convert = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((50, 50)) ])

# dataloader

data = datasets.ImageFolder(root='PetImages/',transform=convert)
Loader = DataLoader(data,batch_size=128,shuffle=True)

# make net
class Net(nn.Module):
    '''
    THIS NN PASSES 
    IMAGE FROM 3 CONV LAYERS , THEN
    IT CONNECTS TO THE FULLY CONN. LAYER
    '''
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.conv3 = nn.Conv2d(16,24, 5)
        self.fc1 = nn.Linear(24*2*2,64)# dim=1 passes data in batches
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,2)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)),kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv3(x)),kernel_size=2)
        x = x.view(-1,24*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=0.01)
EPOCHS=4
TRAIN = False

def Train():
    # TRAINING LOOP
    for i in tqdm(range(EPOCHS)):
        try:
            for img,label in tqdm(Loader):
                img = img.to(device)
                label = label.to(device)
                pred = model(img)
                loss = criterion(pred,label)
                loss.backward()
                optim.step()
                optim.zero_grad()
            if i %1==0:
                print(f'loss:  {loss.item():.4f}')
                print(pred)  
        except Exception as e:
            print(str(e))

    torch.save(model.state_dict(),'catsvdogs.pth')
if TRAIN:
    Train()

TEST = True

def test():
    test = datasets.ImageFolder(root='PetTest/',transform=convert)
    testLoader= DataLoader(test,batch_size=16,shuffle=True)
    checkpoint = torch.load('catsvdogs.pth')
    model.load_state_dict(checkpoint)
    print(model)
    '''
    THERE ARE 13 BATCHES IN TOTAL
    '''
    with torch.no_grad():
        for enum,(img,lab) in enumerate(testLoader):
            CatCount = 0
            DogCount = 0
            outputs = model(img)
            for INDEX_zero,INDEX_one in outputs:
                if INDEX_zero>INDEX_one:
                    CatCount+=1
                elif INDEX_one>INDEX_zero:
                    DogCount+=1
            dogs_acc = (DogCount/sum(lab))*100
            if enum%5==0:
                print(f'Accuracy for Dogs:  {dogs_acc} at batch : {enum}')      
                print(f'DogCount: {DogCount}')
                print(f'sum of dogs : {sum(lab)}')
if TEST:
    test()
'''
THIS NETWORK ALTHOUGH TRAIN AND WORKS PROPERLY 
BUT DOES NOT DO A GOOD WORK IN CLASSIFYING THEM 
AND CRASHES BADLY 
IDK CANT UNDERSTAND WHY ITS NOT LEARNING FROM THE IMAGES  
ITS FAILING VERY POORLY LEARNING TO CLASSIFY 
DOGS VERY CLEARLY THAN  CATS 
OR 
MAY BE ITS JUST SOME ERROR IN THE CODE  
'''
