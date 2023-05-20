import torch
import torch.nn as nn
from torchsummary import summary

from torchmetrics.classification import MulticlassAccuracy

# Device Agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Model Defination
class FedModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.cnn_block = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 20, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = 20, out_channels = 30, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3 ,stride = 2)
    )
    self.fc_block = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = 4320, out_features = 1024),
        nn.ReLU(),
        nn.Linear(in_features = 1024, out_features = 10)
    )

  def forward(self, x):
    conv_x = self.cnn_block(x)
    # print(conv_x.shape)
    fc_x = self.fc_block(conv_x)

    return fc_x
  

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Accuray Function
accuracy_fn = MulticlassAccuracy(num_classes = 10).to(device)