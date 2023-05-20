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
from train_test import train_loop, test_loop, eval_func

# Device Agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# trime train data
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


# trime test data
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

train_subset = torch.utils.data.ConcatDataset([train_data])
test_subset = torch.utils.data.ConcatDataset([test_data])

BATCH_SIZE = 32
train_dataloader = DataLoader(dataset = train_subset, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(dataset = test_subset, batch_size = BATCH_SIZE, shuffle = False)

eval_test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

eval_test_dataloader = DataLoader(dataset = eval_test_dataset, batch_size = BATCH_SIZE, shuffle = False)

# train central model
# init. epochs
epoches = 10

central_model = FedModel().to(device)

optimizer = torch.optim.Adam(params = central_model.parameters(), lr = 1e-3)

central_model_train_loss, central_model_test_loss = [], []
central_model_train_accs, central_model_test_accs = [], []
print()


print("\nCENTRAL BASED MODEL")
print()
start_time = timer()
torch.manual_seed(64)
torch.cuda.manual_seed(64)
for epoch in tqdm(range(epoches)):
#   print(f"Epoch: {epoch+1}")
  train_loss, train_acc, _ = train_loop(model = central_model, dataloader = train_dataloader,
                                    loss_fn = loss_fn, optimizer = optimizer,
                                    accuracy_fn = accuracy_fn, device = device)
  
  test_loss, test_acc = test_loop(model = central_model, dataloader = test_dataloader,
                                  loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                  device = device)
  
  central_model_train_loss.append(train_loss.item())
  central_model_test_loss.append(test_loss.item())
  central_model_train_accs.append(train_acc.item())
  central_model_test_accs.append(test_acc.item())


#   print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Accuray: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
  print("\n")

# plot_graph(model_18_train_loss, model_18_test_loss, model_18_train_accs, model_18_test_accs)

end_time = timer()

print(f"Train Time: {end_time - start_time} seconds.")

# Evalution of model
# init. epochs
epoches = 10

cen_loss, cen_acc = [], []
print()

start_time = timer()
torch.manual_seed(64)
torch.cuda.manual_seed(64)
for epoch in tqdm(range(epoches)):
  print(f"Epoch: {epoch+1}")
  
  eval_loss, eval_acc = eval_func(model = central_model, dataloader = eval_test_dataloader,
                                  loss_fn = loss_fn, accuracy_fn = accuracy_fn,
                                  device = device)
  

  cen_loss.append(eval_loss.item())
  cen_acc.append(eval_acc.item())

  print(f"Eval Loss: {eval_loss:.4f} | Eval Accuracy: {eval_acc:.4f}")
  print()

# plot_graph(model_18_train_loss, model_18_test_loss, model_18_train_accs, model_18_test_accs)

end_time = timer()
print()
print(f"Eval Time: {end_time - start_time} seconds.")

# Confusiom matrix
y_preds = []

confusion_matrix = torch.zeros(10, 10)

with torch.inference_mode():
  for x, y in tqdm(eval_test_dataloader, desc = "Making prediction..."):
    x, y = x.to(device), y.to(device)

    logit = central_model(x)
    pred = torch.softmax(logit.squeeze(), dim = 1).argmax(dim = 1)

    y_preds.append(pred)

y_tensor_preds = torch.cat(y_preds).to('cpu')


confmat = ConfusionMatrix(num_classes = 10, task = 'multiclass')
confmat_tensor = confmat(preds = y_tensor_preds, target = torch.from_numpy(np.array(eval_test_dataset.targets)))

fix, ax = plot_confusion_matrix(conf_mat = confmat_tensor.numpy(), figsize = (10,7))

plt.show()

print(f"\n Overall Accuracy of Central Model: {eval_acc:.4f}")
print("\n")

# Classwise Accuracy
classwise_acc = confmat_tensor.diag()/confmat_tensor.sum(1)
print("\n")
for i in range(len(classwise_acc)):
  print(f"Class {i}: {classwise_acc[i]:.4f}")