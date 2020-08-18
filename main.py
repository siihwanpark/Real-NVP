import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

from dataset import ToyDataset
from torch.utils.data import DataLoader
from model import RealNVP

dataset = ToyDataset('data.csv')

train_loader = DataLoader(dataset=dataset,
                         batch_size=32, 
                         shuffle=True, 
                         num_workers=2)

epochs = 17
k = 4
d = 1
D = 2
hidden_dim = 256

prior = torch.distributions.MultivariateNormal(torch.zeros(D), torch.eye(D))
realNVP = RealNVP(k, d, D, hidden_dim)
optimizer = torch.optim.Adam(realNVP.parameters(), lr = 0.0001)

for epoch in range(epochs):
    for i, x in enumerate(train_loader):
        t = time.time()
        z, log_det_Jacobian = realNVP(x)
        loss = -(prior.log_prob(z) + log_det_Jacobian).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("[%d/%d][%d/%d] loss : %.4f | time :%.2fs"%(epoch+1, epochs, i+1, len(train_loader), loss.item(), time.time()-t))


latent = []
for i, x in enumerate(train_loader):
    z, _ = realNVP(x)
    latent.append(z.detach().numpy())

z = np.concatenate(latent, axis = 0)

fig, ax = plt.subplots()
ax.scatter(z[:,0], z[:,1])
ax.set(title="z = f(X)")
ax.grid()

fig.savefig("f_x.png")
plt.close()