"""
In this TorchDaily we will TRAIN
A MODEL USING TRANSFER LEARNING
Cats Vs Dogs Dataset    

EARLIER ACC==14% OR LESS
NOW ITS 70% AND MORE 
THE POWER OF ALEXNET (PRETRAINED MODELS IS VISIBLE) 
DATE ==> 10-05-21
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
import torchvision
from tqdm import tqdm
import os
import PIL.Image as Image
import time

import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# prepare data
convert = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor(),
    ]
)

# dataloader

data = datasets.ImageFolder(root="PetImages/", transform=convert)
Loader = DataLoader(data, batch_size=64, shuffle=True)

MAP = {0: "Cat", 1: "Dog"}

##UNCOMMENT FOR SEEING THE DATA IMAGES
# fig, ax = plt.subplots(8, 8, figsize=(20, 20))
# fig.suptitle("Dogs And Cats IMages")

# for i, (img, lab) in zip(range(0, 8 * 8), Loader):
#     x = i // 8
#     y = i % 8
#     print(f"{x},{y}")
#     ax[x, y].imshow(img[i].squeeze().permute(1,2,0))
#     ax[x, y].set_title(f"{lab[i]}")
#     ax[x, y].axis("off")
# plt.show()

# # Add on classifier
# # HOW TO CHANGE THE INPUT LAYER WHICH ACCEPTS THE 224*224 INPUT
# # I WANNA CHANGE THAT TO 128*128 THIS SIZE WILL SUFFICE
# We Use AlexNet for transfer learning
##answers below

alexnet = torchvision.models.alexnet(pretrained=True)
for param in alexnet.parameters():
    param.requires_grad = False

# Add a avgpool here
avgpool = nn.AdaptiveAvgPool2d((7, 7))

# Replace the classifier layer
# to customise it according to our output
alexnet.classifier = nn.Sequential(
    nn.Linear(256 * 7 * 7, 1024),
    nn.Linear(1024, 256),
    nn.Linear(256, 2),
)
# putting model in a training mode
alexnet.train()

print(alexnet)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(alexnet.parameters(), lr=0.001)
EPOCHS = 4

TRAIN = False
losses = []


def train_and_validate(model, loss_criterion, optimizer, epochs=25):
    """
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)

    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    """

    start = time.time()
    history = []
    best_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # Set to training mode
        # model.train()

        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(Loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clean existing gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            x = alexnet.features(inputs)
            x = avgpool(x)
            x = x.view(-1, 256 * 7 * 7)
            outputs = alexnet.classifier(x)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)

            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)

            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                # print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

        # Find average training loss and training accuracy
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        epoch_end = time.time()

        print(
            "Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1,
                avg_train_loss,
                avg_train_acc * 100,
                avg_valid_loss,
                avg_valid_acc * 100,
                epoch_end - epoch_start,
            )
        )

        # Save if the model has best accuracy till now
        torch.save(model, "TrainLoopImproveCatsDogs.pth")

    return model, history


if TRAIN:
    trained_model, history = train_and_validate(alexnet, criterion, optimizer, EPOCHS)
    plt.plot(losses)
    plt.show()
    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(["Tr Loss", "Val Loss"])
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.savefig(dataset + "_loss_curve.png")
    plt.show()
    plt.plot(history[:, 2:4])
    plt.legend(["Tr Accuracy", "Val Accuracy"])
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.savefig(dataset + "_accuracy_curve.png")
    plt.show()


TEST = False

history = []


def test():
    test = datasets.ImageFolder(root="PetTest/", transform=convert)
    testLoader = DataLoader(test, batch_size=16, shuffle=False)
    checkpoint = torch.load("catsvdogs.pth")
    alexnet.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for params in alexnet.parameters():
        params.requires_grad == False
    print(alexnet)

    with torch.no_grad():

        # Set to evaluation mode
        alexnet.eval()
        train_data_size = 101
        valid_data_size = 101
        # Validation loop
        # Loss and Accuracy within the epoch
        valid_loss = 0.0
        valid_acc = 0.0
        for j, (inputs, labels) in enumerate(testLoader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass - compute outputs on input data using the model
            x = alexnet.features(inputs)
            x = avgpool(x)
            x = x.view(-1, 256 * 7 * 7)
            outputs = alexnet.classifier(x)

            # Compute loss
            loss = criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            valid_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            valid_acc += acc.item() * inputs.size(0)

            print(
                """Validation Batch number: {:03d},
                 Validation: Loss: {:.4f},
                  Accuracy: {:.4f}""".format(
                    j, loss.item(), acc.item()
                )
            )

    # Find average training loss and training accuracy
    avg_valid_loss = valid_loss / valid_data_size
    avg_valid_acc = valid_acc / valid_data_size

    history.append([avg_valid_loss, avg_valid_acc])
    print(
        " Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%".format(
            avg_train_loss,
            avg_train_acc * 100,
            avg_valid_loss,
            avg_valid_acc * 100,
        )
    )
    plt.plot(valid_acc)
    plt.plot(valid_loss)
    plt.show()


if TEST:
    test()
    print("Validation Complete")
    with open("ModelHistory.txt", "w") as f:
        for i in history:
            f.writelines(f"{i}")
    print("Validation Complete")

## This model reported a accuracy of 97%(on DOGS ONLY) using AlexNet
## the Pros of using a pretrained model is clearly seen here
## date -- 13th April 2021 (thursday)
####ACCURACY AND OTHER THINGS TOO TO BE APPENDED SOON ######


PREDICT = True


def predict(model, test_image_name):
    """
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    """
#     try:
    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )
    test_image = Image.open(test_image_name)
    test_image_tensor = transform(test_image).to(device)
    plt.imshow(test_image)
    plt.axis('off')
    plt.imshow(test_image_tensor.cpu().squeeze().permute(1, 2, 0))
    plt.show()

    with torch.no_grad():
        model.eval()
        test_image_tensor = test_image_tensor.unsqueeze(0)
        print(test_image_tensor.shape)
        x = alexnet.features(test_image_tensor)
        x = avgpool(x)
        x = x.view(-1, 256 * 7 * 7)
        out = alexnet.classifier(x)
        ###THESE ARE SCORES OF THE ACC. ###
        ### UNCOMMENT TO SEE THE SCORES OF EACH CLASS ###
#         ps = torch.exp(out)
#         print(f'ps:  {ps}')
#         topk, topclass = ps.topk(2, dim=1)
#         print(f'ps.topk:  {ps.topk(2, dim=1)}')
#         print(f'topclass:  {topclass}')
        print(
            "Predcition",
            MAP[out.numpy().argmax()]
            )

        # print(f"out: {out.numpy().argmax()}")
#     except Exception as e:
#         print(e)

if PREDICT:
    checkpoint = torch.load("ImprovedCatVsDogsModel.pth",map_location=torch.device('cpu'))
    alexnet.load_state_dict(checkpoint["state_dict"])
    alexnet = alexnet.to(device)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for params in alexnet.parameters():
        params.requires_grad == False
    predict(alexnet, 'PetTest/CatTest.jpg')
