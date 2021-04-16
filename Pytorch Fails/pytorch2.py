import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import logging
logging.basicConfig(filename='NET1.log',format='%(asctime)s - %(message)s',level=logging.INFO)

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="../data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size=64

train_loader = DataLoader(training_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)

for x,y in test_loader:
    print(f'xshape:{x.shape}')
    print(f'yshape:{y.shape},{y.dtype}')
    break

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'running onto--{device}')

#define model

class NET(nn.Module):
    def __init__(self):
        super(NET,self).__init__()
        self.flatten = nn.Flatten() #28 by 28 dim sqeezed down to 28*28 by 1
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,10),
            nn.ReLU(),
            # nn.Linear(64,32),
            # nn.ReLU(),
            # nn.Linear(32,10),
            # nn.ReLU(),
            )
        
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        
model = NET().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

def train(loader,model,loss_fn,optimizer):
    size=len(loader)
    
    for batch,(x,y) in enumerate(loader):
        x,y = x.to(device),y.to(device) # this is usually considered a bad approach
        # x,y  = torch.tensor(x,device=device),torch.tensor(y,device=device) is much more powerful
        
        # cal preds err
        preds = model(x)
        loss = loss_fn(preds,y)

        #backprop
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0 :
            loss,current = loss.item(),batch*len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # logging.INFO('loss is --{} '.format(loss))

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model)



torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


print("Done!")
 
