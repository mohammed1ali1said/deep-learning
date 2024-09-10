import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys

N, D_in, H, D_out = 64, 1000, 100, 10

#Data
x=Variable(torch.randn(N, D_in), requires_grad=False)
y=Variable(torch.randn(N, D_out), requires_grad=False)

#Our network graph
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)
lr = 1e-6
loss_func = torch.nn.MSELoss(reduction='sum')
opt = torch.optim.Adam(model.parameters(), lr=lr)


for t in range(0,5001):
  # Apply the layers
  y_pred = model(x)
  loss = loss_func(y_pred, y)

  # Print progress
  if (t%100==0):
    print(t, end="  ")
    print(loss.data.item())

  # Zero the gradients
  model.zero_grad()

  # Compute gradient
  loss.backward()

  #Update weights

  for param in model.parameters():
    param.data -= lr * param.grad.data
