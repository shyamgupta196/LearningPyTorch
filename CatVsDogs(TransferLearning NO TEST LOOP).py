'''
In this TorchDaily we will TRAIN
A MODEL USING TRANSFER LEARNING
Cats Vs Dogs Dataset    

EARLIER ACC==14% OR LESS
NOW ITS 70% AND MORE 
THE POWER OF ALEXNET (PRETRAINED MODELS IS VISIBLE) 
DATE ==> 10-05-21
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms,datasets,models
import torchvision
from tqdm import tqdm
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# prepare data
convert = transforms.Compose(
    [transforms.Resize((128,128)),transforms.RandomHorizontalFlip(0.2),transforms.ToTensor() ])

# dataloader

data = datasets.ImageFolder(root='PetImages/',transform=convert)
Loader = DataLoader(data,batch_size=64,shuffle=True)


##UNCOMMENT FOR SEEING THE DATA IMAGES    
fig,ax = plt.subplots(8,8,figsize=(20,20))
fig.suptitle('Dogs And Cats IMages')

for i,(img,lab) in zip(range(0,8*8),Loader):
    x = i//8
    y = i%8
    print(f'{x},{y}')
    ax[x,y].imshow(img)
    ax[x,y].set_title(f'{lab[i]}')
    ax[x,y].axis('off')
plt.show()
 
# # Add on classifier
# # HOW TO CHANGE THE INPUT LAYER WHICH ACCEPTS THE 224*224 INPUT 
# # I WANNA CHANGE THAT TO 128*128 THIS SIZE WILL SUFFICE 
# We Use AlexNet for transfer learning

class OurAlex(nn.Module):

    def __init__(self, num_classes=8):
        super(OurAlex, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        for param in self.alexnet.parameters():
            param.requires_grad = False

        # Add a avgpool here
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Replace the classifier layer
        # to customise it according to our output
        self.alexnet.classifier = nn.Sequential(
            nn.Linear(256*7*7, 1024),
            nn.Linear(1024,256),
            nn.Linear(256,num_classes))
        

    def forward(self, x):
        x = self.alexnet.features(x)
        x = self.avgpool(x)
        x = x.view(-1,256*7*7)
        x = self.alexnet.classifier(x)
        return x

model = OurAlex(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
EPOCHS = 1

TRAIN = False
losses = []
def training_loop(model,optimizer,epochs):
    for epoch in range(epochs):
        try:
            for img,lab in tqdm(Loader):
                img = img.to(device)
                lab = lab.to(device)
                predictions = model(img)
                loss = criterion(predictions,lab)
                optimizer.step()
                optimizer.zero_grad()        
                torch.save(model,'catsvdogs.pth')    
                losses.append(loss.item())
                print(f'loss:  {loss.item():.4f}')
        except Exception as e:
            print(str(e))
if TRAIN:
    training_loop(model,optimizer,EPOCHS)








'''
THIS TEST LOOP CONSISTS OF BUGS 
HENCE DO NOT USE THIS WE WILL USE SOME 
OTHER FILE TO DO THE SAME 
'''

# TEST = True
# def test():
#     test = datasets.ImageFolder(root='PetTest/',transform=convert)
#     testLoader= DataLoader(test,batch_size=16,shuffle=True)
#     checkpoint = torch.load('catsvdogs.pth')
#     model.load_state_dict(checkpoint)
#     print(model)
#     '''
#     THERE ARE 13 BATCHES IN TOTAL
#     '''
#     with torch.no_grad():
#         for enum,(img,lab) in enumerate(testLoader):
#             CatCount = 0
#             DogCount = 0
#             outputs = model(img)
#             for INDEX_zero,INDEX_one in outputs:
#                 if INDEX_zero>INDEX_one:
#                     CatCount+=1
#                 elif INDEX_one>INDEX_zero:
#                     DogCount+=1
#             dogs_acc = (DogCount/sum(lab))*100
#             if enum%5==0:
#                 print(f'Accuracy for Dogs:  {dogs_acc} at batch : {enum}')      
#                 print(f'DogCount: {DogCount}')
#                 print(f'sum of dogs : {sum(lab)}')
# if TEST:
#     test()


# I DID IT FOR THE FIRST TIME TODAY 
# TRAINED THE MODEL USING TRANSFER LEARNING MODEL (AlexNet)  
# DATE -- 10-05-2021
