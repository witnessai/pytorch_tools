import torch
from torchvision import models
import ipdb

model = models.resnet50(pretrained=False)  # if true, it will take a lot of time for procedure to download pretrained model.
ipdb.set_trace()

# print full model
# print(model)

# view one convolution layer of model, e.g.
# input:
# print(model.conv1)
# output:
# Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# view parameters of model
# model.parameters() is a generator
# for parameter in model.parameters():
#     print(parameter)

# view all names of model parameters
# for i in model.state_dict():
#     print(i)


# view all names and values of model parameters
# the output will tell us whether the parameter is require gradient or not in the last line
# for i in model.named_parameters():
#     print(i)


