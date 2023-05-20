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

from model import FedModel, accuracy_fn, loss_fn
from train_test import eval_func
from Clients.client1 import get_client1
from Clients.client2 import get_client2
from Clients.client3 import get_client3

# Device Agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get dataset
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
])

eval_test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

BATCH_SIZE = 32

eval_test_dataloader = DataLoader(dataset = eval_test_dataset, batch_size = BATCH_SIZE, shuffle = False)


# get avg weights
def get_avg_weight(clinets_model):
  cnn_1_weight = torch.zeros(size = clinets_model[0].cnn_block[0].weight.shape)
  cnn_1_bias = torch.zeros(size = clinets_model[0].cnn_block[0].bias.shape)

  cnn_2_weight = torch.zeros(size = clinets_model[0].cnn_block[2].weight.shape)
  cnn_2_bias = torch.zeros(size = clinets_model[0].cnn_block[2].bias.shape)

  cnn_3_weight = torch.zeros(size = clinets_model[0].cnn_block[4].weight.shape)
  cnn_3_bias = torch.zeros(size = clinets_model[0].cnn_block[4].bias.shape)

  fc_1_weight = torch.zeros(size = clinets_model[0].fc_block[1].weight.shape)
  fc_1_bias = torch.zeros(size = clinets_model[0].fc_block[1].bias.shape)

  fc_2_weight = torch.zeros(size = clinets_model[0].fc_block[3].weight.shape)
  fc_2_bias = torch.zeros(size = clinets_model[0].fc_block[3].bias.shape)


  with torch.inference_mode():
    for model in clients_model:
      cnn_1_weight += model.cnn_block[0].weight.data.clone()
      cnn_1_bias += model.cnn_block[0].bias.data.clone()

      cnn_2_weight += model.cnn_block[2].weight.data.clone()
      cnn_2_bias += model.cnn_block[2].bias.data.clone()

      cnn_3_weight += model.cnn_block[4].weight.data.clone()
      cnn_3_bias += model.cnn_block[4].bias.data.clone()

      fc_1_weight += model.fc_block[1].weight.data.clone()
      fc_1_bias += model.fc_block[1].bias.data.clone()

      fc_2_weight += model.fc_block[3].weight.data.clone()
      fc_2_bias += model.fc_block[3].bias.data.clone()

    cnn_1_weight = cnn_1_weight/len(clients_model)
    cnn_1_bias = cnn_1_bias/len(clients_model)

    cnn_2_weight = cnn_2_weight/len(clients_model)
    cnn_2_bias = cnn_2_bias/len(clients_model)

    cnn_3_weight = cnn_3_weight/len(clients_model)
    cnn_3_bias = cnn_3_bias/len(clients_model)

    fc_1_weight = fc_1_weight/len(clients_model)
    fc_1_bias = fc_1_bias/len(clients_model)

    fc_2_weight = fc_2_weight/len(clients_model)
    fc_2_bias = fc_2_bias/len(clients_model)


  return cnn_1_weight, cnn_1_bias, cnn_2_weight, cnn_2_bias, cnn_3_weight, cnn_3_bias, fc_1_weight, fc_1_bias, fc_2_weight, fc_2_bias


# update weights
def update_fedmodel_weights(model, clinets_model):
  cnn_1_weight, cnn_1_bias, cnn_2_weight, cnn_2_bias, cnn_3_weight, cnn_3_bias, fc_1_weight, fc_1_bias, fc_2_weight, fc_2_bias = get_avg_weight(clinets_model)

  with torch.inference_mode():
    model.cnn_block[0].weight.data = cnn_1_weight.data.clone()
    model.cnn_block[0].bias.data = cnn_1_bias.data.clone()

    model.cnn_block[2].weight.data = cnn_2_weight.data.clone()
    model.cnn_block[2].bias.data = cnn_2_bias.data.clone()

    model.cnn_block[4].weight.data = cnn_3_weight.data.clone()
    model.cnn_block[4].bias.data = cnn_3_bias.data.clone()

    model.fc_block[1].weight.data = fc_1_weight.data.clone()
    model.fc_block[1].bias.data = fc_1_bias.data.clone()

    model.fc_block[3].weight.data = fc_2_weight.data.clone()
    model.fc_block[3].bias.data = fc_2_bias.data.clone()

  return model


print("\nFedrated Model")
# get model weights from clients
client_1_model, client_1_loss_fn, client_1_optimizer = get_client1()
client_2_model, client_2_loss_fn, client_2_optimizer = get_client2()
client_3_model, client_3_loss_fn, client_3_optimizer = get_client3()

clients_model = []
clients_model.append(client_1_model)
clients_model.append(client_2_model)
clients_model.append(client_3_model)


# fedrated model
fed_model = FedModel().to(device)
fed_model = update_fedmodel_weights(fed_model, clients_model).to(device)

# Eval Fedrated model
# init. epochs
epoches = 10

fed_loss, fed_acc = [], []
print("\n Eval Fedrated model")

start_time = timer()
torch.manual_seed(64)
torch.cuda.manual_seed(64)
for epoch in tqdm(range(epoches)):
  print(f"Epoch: {epoch+1}")
  
  eval_loss, eval_acc = eval_func(model = fed_model, dataloader = eval_test_dataloader,
                                  loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                  device = device)
  

  fed_loss.append(eval_loss.item())
  fed_acc.append(eval_acc.item())

  print(f"Eval Loss: {eval_loss:.4f} | Eval Accuracy: {eval_acc:.4f}")
  print()

# plot_graph(model_18_train_loss, model_18_test_loss, model_18_train_accs, model_18_test_accs)

end_time = timer()
print("\n")
print(f"Eval Time: {end_time - start_time} seconds.")

# Confusion matrix
y_preds = []


with torch.inference_mode():
  for x, y in tqdm(eval_test_dataloader, desc = "Making prediction..."):
    x, y = x.to(device), y.to(device)

    logit = fed_model(x)
    pred = torch.softmax(logit.squeeze(), dim = 1).argmax(dim = 1)

    y_preds.append(pred)

y_tensor_preds = torch.cat(y_preds).to('cpu')

confmat = ConfusionMatrix(num_classes = 10, task = 'multiclass')
confmat_tensor = confmat(preds = y_tensor_preds, target = torch.from_numpy(np.array(eval_test_dataset.targets)))

fix, ax = plot_confusion_matrix(conf_mat = confmat_tensor.numpy(), figsize = (10,7))

plt.show()

print(f"\n Overall Accuracy of Fedrated Model: {eval_acc:.4f}")
print("\n")
# Classwise accuray
classwise_acc = confmat_tensor.diag()/confmat_tensor.sum(1)

for i in range(len(classwise_acc)):
  print(f"Class {i}: {classwise_acc[i]:.4f}")