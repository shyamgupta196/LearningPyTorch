'''
Transforms 
author == 'shyamgupta196'
python.__version__ == 3.8.3
'''
# %%
# In order to script the transformations, please use torch.nn.Sequential instead of Compose.




WE ARE LEAVING THE TOPIC OF TRANSFORMATIONS FOR THE MOMENTS
AND WILL COMEBACK LATER ,WHEN I WILL HAVE A BETTER UNDERSTANDING 
OF THE FRAMEWORK
 


import PIL.Image as image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image

# plt.imshow(Transforms(image.open('../data/img1.jpg')))
# plt.show()

# JUST CONV TENSOR TO FLOAT
# ValueError: std evaluated to zero after
# conversion to torch.uint8, leading to division by zero.
# if __name__ == '__main__':
def Transform(img):
    return transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
