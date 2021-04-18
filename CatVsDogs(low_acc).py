'''
In this TorchDaily we will see a 
simple convlution network using

Cats Vs Dogs Dataset    
'''
import PIL.Image as image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from tqdm import tqdm
import  torch.nn.functional as F
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# prepare data
convert = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((50, 50)) ])

# dataloader

data = datasets.ImageFolder(root='PetImages/',transform=convert)
Loader = DataLoader(data,batch_size=64,shuffle=False)

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
        self.fc1 = nn.Linear(24*8*8,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,2)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)),kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv3(x)),kernel_size=2)
        x = x.view(-1,24*8*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=0.01)
EPOCHS=5

TRAIN = True

def Train():
    # checkpoint = torch.load('catsvdogs.pth')
    # model.load_state_dict(checkpoint)
    # TRAINING LOOP
    for i in tqdm(range(EPOCHS)):
        try:
            for img,label in tqdm(Loader):
                img = img.to(device)
                label = label.to(device)
                pred = model(img)
                loss = criterion(pred,label)
                optim.zero_grad()
                loss.backward()
                optim.step()
                torch.save(model.state_dict(),'catsvdogs.pth')
            if i %1==0:
                print(f'loss:  {loss.item():.4f}')
                print(pred)  
        except Exception as e:
            print(str(e))

if TRAIN:
    Train()

TEST = False

def test():
    '''
    I Still Have To Learn Better ways of scoring model accuracy
    I Guess I made a mistake here !!
    '''
    test = datasets.ImageFolder(root='PetTest/',transform=convert)
    testLoader= DataLoader(test,batch_size=16,shuffle=False)
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
            lab_list = lab.numpy()
            labels_counter = Counter(lab_list)
            outputs = model(img)
            for INDEX_zero,INDEX_one in outputs:
                INDEX_one = INDEX_one.round()
                INDEX_zero = INDEX_zero.round()
                print(INDEX_zero,INDEX_one)
                if INDEX_zero==-1:
                    CatCount+=1
                elif INDEX_one>INDEX_zero:
                    DogCount+=1
            print(f'CatCount: {CatCount}')
            print(f'DogCount:  {DogCount}')
                
            if enum%3==0:
                '''
                    there are 100 images for dogs and cats resp.
                    it was not learning that well so i have to
                    make some custom changes considering batch size

                    I'll improve this soon !!
                '''
                if enum<7:
                    cats_acc = (CatCount/int(labels_counter[0]))*100
                    print(f'Cats acc = {cats_acc}')
                    print(f'CatsCount: {CatCount}')
                else:
                    dogs_acc = (DogCount/int(labels_counter[1]))*100
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
May be the architecture is not good at learning features
'''
