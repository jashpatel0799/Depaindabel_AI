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


train_dataset = datasets.SVHN(root = './data', split = "train", transform = transforms.ToTensor(), download = True)
test_dataset = datasets.SVHN(root = './data', split = "test", transform = transforms.ToTensor(), download = True)

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

client3_train_subset = torch.utils.data.ConcatDataset([train_data])
client3_test_subset = torch.utils.data.ConcatDataset([test_data])

BATCH_SIZE = 32
client3_train_dataloader = DataLoader(dataset = client3_train_subset, batch_size = BATCH_SIZE, shuffle = True)
client3_test_dataloader = DataLoader(dataset = client3_test_subset, batch_size = BATCH_SIZE, shuffle = False)

client_3_model = FedModel().to(device)
client_3_loss_fn = nn.CrossEntropyLoss()

def update_client(client3_train_dataloader, client3_test_dataloader, client_3_model, client_3_loss_fn, accuracy_fn, device):
    print("\nCLIENT 3:")
    # load model
    if os.path.exists('weights/client3.pth'): 
        MODEL_NAME = "client3.pth"
        MODEL_PATH_SAVE = "weights/" + MODEL_NAME

        if torch.cuda.is_available() == False:
            client_3_model.load_state_dict(torch.load(f = MODEL_PATH_SAVE, map_location=torch.device('cpu')))
        else:
            client_3_model.load_state_dict(torch.load(f = MODEL_PATH_SAVE))

    client_3_optimizer = torch.optim.Adam(params = client_3_model.parameters(), lr = 1e-3)

    client_3_model, client_3_loss_fn, client_3_optimizer, test_acc = train_client(client_3_model, client3_train_dataloader, client3_test_dataloader,
                                                                        client_3_optimizer, client_3_loss_fn, accuracy_fn, device = device)

    print()

    # save model
    MODEL_PATH = Path("weights")
    MODEL_PATH.mkdir(parents = True, exist_ok = True)

    MODEL_NAME = "client3.pth"
    MODEL_PATH_SAVE = MODEL_PATH / MODEL_NAME

    print(f"model saved at: {MODEL_PATH_SAVE}")
    if os.path.exists('weights/client3.pth'):
        os.remove("weights/client3.pth")
        torch.save(obj = client_3_model.state_dict(), f = MODEL_PATH_SAVE)
    else:
        torch.save(obj = client_3_model.state_dict(), f = MODEL_PATH_SAVE)

    # Confusion Matrix
    y_preds = []


    with torch.inference_mode():
        for x, y in tqdm(client3_test_dataloader, desc = "Making prediction..."):
            x, y = x.to(device), y.to(device)

            logit = client_3_model(x)
            pred = torch.softmax(logit.squeeze(), dim = 1).argmax(dim = 1)

            y_preds.append(pred)

    y_tensor_preds = torch.cat(y_preds).to('cpu')


    target = [j for (i,j) in client3_test_subset]

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

    return  client_3_model, client_3_loss_fn, client_3_optimizer


def get_client3():
   global client3_train_dataloader, client3_test_dataloader, client_3_model, client_3_loss_fn, accuracy_fn, device

   client_3_model, client_3_loss_fn, client_3_optimizer = update_client(client3_train_dataloader, client3_test_dataloader, client_3_model, client_3_loss_fn, accuracy_fn, device)

   return  client_3_model, client_3_loss_fn, client_3_optimizer