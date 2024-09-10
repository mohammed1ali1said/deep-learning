import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


EPOCHS_TO_TRAIN = 120
D_in=2
H=3
D_out = 1
lr=0.1

seed = 74
torch.manual_seed(seed)

model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out)
)

net = model

inputs = Variable(torch.Tensor([[0,0],[0,1],[1,0],[1,1]]), requires_grad=False)
targets = Variable(torch.Tensor([0,1,1,0]), requires_grad=False)
targets = targets.view(-1,1,)


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr)

print("Training loop:")
for idx in range(0, EPOCHS_TO_TRAIN):

    #net step
    optimizer.zero_grad()
    output = net(inputs)

    loss = criterion(output, targets)
    loss.backward()

    #update
    optimizer.step()



print("")
print("Final results:")
acc=0
for input, target in zip(inputs, targets):
    output = net(input)
    pred = int(round(output.data.numpy()[0]))
    y = int(target.data.numpy()[0])
    if pred==y: acc+=1/4
    print("Input:[{},{}] Target:[{}] Predicted:[{}] Error:[{}]".format(
        int(input.data.numpy()[0]),
        int(input.data.numpy()[1]),
        int(target.data.numpy()[0]),
        round(output.data.numpy()[0]),
        round(float(abs(target.data.numpy()[0] - round(output.data.numpy()[0]))))
    ))


print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("Accuracy: {}".format(acc*100))
