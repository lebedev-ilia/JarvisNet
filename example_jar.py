import torch

x = torch.FloatTensor(1, 2)

print(x)

z = torch.zeros(1, 2)

print(z)

c = torch.cat((z, x), dim=1)

print(c)