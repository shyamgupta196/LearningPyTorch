'''
DATASET'S AND DATALOADER'S

AUTHOR - shyamgupta196
PYTHON - 3.8.3
'''
ON HOLD 

import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from pytorch4 import Transform
import torch.nn as nn
from torchvision import transforms

# Transforms = nn.Sequential(
#     transforms.CenterCrop(10),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#     )


training_data = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=Transform
)

test_data = datasets.FashionMNIST(
    root="../data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img, cmap="gray")
plt.show()

import pandas as pd
import os


# class CustomDataSet(Dataset):
#     def __init__(self,annotations,root_dir,transform):
#         self.data = pd.read_csv(annotations)
#         self.root_dir = root_dir
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.data)
    
    
#     def __getitem__(self,idx):
#         image_path = os.path.join(self.root_dir,self.data.iloc[idx,0])
#         image = read_image(image_path)
#         label = self.data.iloc[idx,1]
        
#         if transform:
#             image = self.transform(image)
#         sample = {'image':image,'label':labels_map.get(label)}
        
#         return image,label

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[10].squeeze()

# I MAPPED THE LABELS ADDITIONALY
label = labels_map.get(int(train_labels[10]))
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")


