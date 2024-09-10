

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys

N = 1000
D_in=2
H=4
D_out = 1

#Data
x=Variable(torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32), requires_grad=False)
y=Variable(torch.tensor([[0],[1],[1],[0]], dtype=torch.float32), requires_grad=False)

#Our network graph
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out)
)

# loss_func = torch.nn.MSELoss(reduction='sum')
loss_func = F.binary_cross_entropy_with_logits
lr = 1e-2
opt = torch.optim.Adam(model.parameters(), lr=lr)

itr = 2000
for t in range(0,itr+1):
  # Apply the layers

  y_pred = model(x)
  # print(y.shape)
  # print(y_pred.shape)
  # y = torch.round(y)
  loss = loss_func(y_pred, y)

  # Print progress
  if t % 200 == 0:
    print(t, end="  ")
    print(loss.data.item())

  # Zero the gradients
  model.zero_grad()

  # Compute gradient
  loss.backward()

  #Update weights
  opt.step()
