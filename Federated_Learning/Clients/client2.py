import sys
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import torch.utils.data 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from timeit import default_timer as timer
from tqdm.auto import tqdm



# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# print(parent)

from model import FedModel, accuracy_fn
from train_test import train_client

# Device Agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'


import random

# Define a function to randomly assign a color to each digit
def random_color(x):
    r = random.uniform(0.0, 1.0)
    g = random.uniform(0.0, 1.0)
    b = random.uniform(0.0, 1.0)
    color = torch.tensor([r, g, b])
    return color.view(3, 1, 1).repeat(1, x.shape[1], x.shape[2])

# Define the transform for the dataset
color_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the images to 32x32
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Replicate the grayscale channel three times
    transforms.Lambda(lambda x: x * random_color(x))  # Randomly assign a color to each digit
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=color_transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=color_transform, download=True)

class_count = [0 for i in range(10)]
train_count = 0
i = 0
train_data = []
while train_count != 10000:
  image = train_dataset[i][0]
  label = train_dataset[i][1]

  if class_count[label] >= 1000:
    i += 1
    continue
  else:
    train_data.append((image, label))
    class_count[label] += 1
    train_count += 1
    i += 1
  # print(f"count: {train_count}")


class_count = [0 for i in range(10)]
test_count = 0
i = 0
test_data = []
while test_count != 5000:
  image = test_dataset[i][0]
  label = test_dataset[i][1]

  if class_count[label] >= 500:
    i += 1
    continue
  else:
    test_data.append((image, label))
    class_count[label] += 1
    test_count += 1
    i += 1
  # print(f"count: {test_count}")

client2_train_subset = torch.utils.data.ConcatDataset([train_data])
client2_test_subset = torch.utils.data.ConcatDataset([test_data])

BATCH_SIZE = 32
client2_train_dataloader = DataLoader(dataset = client2_train_subset, batch_size = BATCH_SIZE, shuffle = True)
client2_test_dataloader = DataLoader(dataset = client2_test_subset, batch_size = BATCH_SIZE, shuffle = False)


client_2_model = FedModel().to(device)
client_2_loss_fn = nn.CrossEntropyLoss()

def update_client(client2_train_dataloader, client2_test_dataloader, client_2_model, client_2_loss_fn, accuracy_fn, device):
    print("\nCLIENT 2:")
    # load model
    if os.path.exists('weights/client2.pth'): 
        MODEL_NAME = "client2.pth"
        MODEL_PATH_SAVE = "weights/" + MODEL_NAME

        if torch.cuda.is_available() == False:
            client_2_model.load_state_dict(torch.load(f = MODEL_PATH_SAVE, map_location=torch.device('cpu')))
        else:
            client_2_model.load_state_dict(torch.load(f = MODEL_PATH_SAVE))

    client_2_optimizer = torch.optim.Adam(params = client_2_model.parameters(), lr = 1e-3)

    client_2_model, client_2_loss_fn, client_2_optimizer, test_acc = train_client(client_2_model, client2_train_dataloader, client2_test_dataloader,
                                                                        client_2_optimizer, client_2_loss_fn, accuracy_fn, device = device)
    print()

    # save model
    MODEL_PATH = Path("weights")
    MODEL_PATH.mkdir(parents = True, exist_ok = True)

    MODEL_NAME = "client2.pth"
    MODEL_PATH_SAVE = MODEL_PATH / MODEL_NAME

    print(f"model saved at: {MODEL_PATH_SAVE}")
    if os.path.exists('weights/client2.pth'):
        os.remove("weights/client2.pth")
        torch.save(obj = client_2_model.state_dict(), f = MODEL_PATH_SAVE)
    else:
        torch.save(obj = client_2_model.state_dict(), f = MODEL_PATH_SAVE)


    # Confusiom matrix
    y_preds = []

    with torch.inference_mode():
        for x, y in tqdm(client2_test_dataloader, desc = "Making prediction..."):
            x, y = x.to(device), y.to(device)

            logit = client_2_model(x)
            pred = torch.softmax(logit.squeeze(), dim = 1).argmax(dim = 1)

            y_preds.append(pred)

    y_tensor_preds = torch.cat(y_preds).to('cpu')


    target = [j for (i,j) in client2_test_subset]

    confmat = ConfusionMatrix(num_classes = 10, task = 'multiclass')
    confmat_tensor = confmat(preds = y_tensor_preds, target = torch.from_numpy(np.array(target)))

    fix, ax = plot_confusion_matrix(conf_mat = confmat_tensor.numpy(), figsize = (10,7))

    plt.show()

    print(f"\n Overall Accuracy of Client Model: {test_acc:.4f}")
    print("\n")
    
    # Classwise Accuracy
    classwise_acc = confmat_tensor.diag()/confmat_tensor.sum(1)

    for i in range(len(classwise_acc)):
        print(f"Class {i}: {classwise_acc[i]:.4f}")

    return client_2_model, client_2_loss_fn, client_2_optimizer


def get_client2():
   global client2_train_dataloader, client2_test_dataloader, client_2_model, client_2_loss_fn, accuracy_fn, device

   client_2_model, client_2_loss_fn, client_2_optimizer = update_client(client2_train_dataloader, client2_test_dataloader, client_2_model, client_2_loss_fn, accuracy_fn, device)

   return  client_2_model, client_2_loss_fn, client_2_optimizer