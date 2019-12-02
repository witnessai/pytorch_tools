import torch
import torch.nn as nn
from torchvision import models


class B(torch.nn.Module):
    def __init__(self):
        super(B, self).__init__()
        self.fc = nn.Linear(100, 100)
    def forward(self, input):
        x = self.fc(input)
        return x
modeltwo = B()

class A(torch.nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.fc1 = nn.Linear(100, 100)
        # self.fc2 = modeltwo
    def forward(self, input):
        x = self.fc1(input)
        x = modeltwo.forward(x)
        return x




modelone = A()

input = torch.rand(10, 100)
output = modelone(input)
print(output)
