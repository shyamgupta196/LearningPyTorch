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
EPOCHS = 1

TRAIN = False
losses = []


def training_loop(model, optimizer, epochs):
    for epoch in range(epochs):
        try:
            for img, lab in tqdm(Loader):
                img = img.to(device)
                lab = lab.to(device)
                x = alexnet.features(img)
                x = avgpool(x)
                x = x.view(-1, 256 * 7 * 7)
                predictions = alexnet.classifier(x)
                loss = criterion(predictions, lab)
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
                print(f"loss:  {loss.item():.4f}")
        except Exception as e:
            print(str(e))
        state = {
            "epoch": epoch,
            "state_dict": alexnet.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, "catsvdogs.pth")


if TRAIN:
    training_loop(alexnet, optimizer, EPOCHS)
    plt.plot(losses)
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
                  Accuracy: {:.4f}""".format(j,
                   loss.item(),
                    acc.item()
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
    transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )
    test_image = Image.open(test_image_name)
    test_image_tensor = transform(test_image).to(device)
    plt.imshow(test_image)
    plt.imshow(test_image_tensor.squeeze().permute(1, 2, 0))
    plt.show()

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        test_image_tensor = test_image_tensor.unsqueeze(0)
        print(test_image_tensor.shape)
        x = alexnet.features(test_image_tensor)
        x = avgpool(x)
        x = x.view(-1, 256 * 7 * 7)
        out = alexnet.classifier(x)

        ps = torch.exp(out)
        topk, topclass = ps.topk(2, dim=1)
        for i in range(2):
            print(
                "Predcition",
                i + 1,
                ":",
                f"topclass {topclass.numpy()}",
                MAP[topclass.numpy()[0][i]],
                ", Score: ",
                f"topk {topk.numpy()}",
                topk.numpy()[0][i],
            )

        print(f"out: {out}")


if PREDICT:
    checkpoint = torch.load("catsvdogs.pth")
    alexnet.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for params in alexnet.parameters():
        params.requires_grad == False
    predict(alexnet, "PetTest/CatTest.jpg")
