#!/usr/bin/env python
# coding: utf-8

# ## MAD GAN

# In[15]:


# Initialization of libraries
import torch
import torch.nn
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import random
from random import randint
from sklearn.mixture import GaussianMixture
from transformer import DataTransformer
import pandas as pd
device = torch.device('cpu')
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


# defining parameters for the training
mb_size = 2 # Batch Size
Z_dim = 11  # Length of noise vector
X_dim = 11  # Input Length
h_dim = 128  # Hidden Dimension
lr = 1e-2    # Learning Rate
# num_gen = 4


# In[3]:


# np.random.seed(1)
# gmm = GaussianMixture(n_components=5, covariance_type='spherical')
# gmm.means_ = np.array([[10], [20], [60], [80], [110]])
# gmm.covariances_ = np.array([[3], [3], [2], [2], [1]]) ** 2
# gmm.weights_ = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# X = gmm.sample(2000)
# data = X[0]
# data = (data - min(X[0]))/(max(X[0])-min(X[0]))
# plt.hist(data, 200000, density=False, histtype='stepfilled', alpha=1)


train_data = pd.read_csv('data/testsingle.csv')
transformer = DataTransformer()
discrete_columns=tuple()
num_gen = train_data.shape[1]-1
transformer.fit(train_data, discrete_columns)
train_data = transformer.transform(train_data)


# In[4]:

class formerNet(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        self.former = torch.nn.Sequential(
        torch.nn.Linear(Z_dim, h_dim),
        torch.nn.BatchNorm1d(h_dim),
        torch.nn.PReLU()
        )
    def forward(self,x):
        x = self.former(x)             # linear output
        return x

class latterNet(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        self.latter = torch.nn.Sequential(
        formerNet(),
        torch.nn.Linear(h_dim, h_dim),
        torch.nn.BatchNorm1d(h_dim),
        torch.nn.PReLU(),
        torch.nn.Linear(h_dim, X_dim),
        torch.nn.Sigmoid()
        )
        # self.hidden = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer
        # self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = self.latter(x)
        return x

G = []
for i in range(num_gen):
    G.append(latterNet().cpu())
    # G.append(torch.nn.Sequential(
    #     torch.nn.Linear(Z_dim, h_dim),
    #     torch.nn.BatchNorm1d(h_dim),
    #     torch.nn.PReLU(),
    #     torch.nn.Linear(h_dim, h_dim),
    #     torch.nn.BatchNorm1d(h_dim),
    #     torch.nn.PReLU(),
    #     torch.nn.Linear(h_dim, X_dim),
    #     torch.nn.Sigmoid()
    # ).cpu())

D = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim),
    torch.nn.BatchNorm1d(h_dim),
    torch.nn.LeakyReLU(0.2),
    torch.nn.Linear(h_dim, h_dim),
    torch.nn.BatchNorm1d(h_dim),
    torch.nn.LeakyReLU(0.2),
    torch.nn.Linear(h_dim, num_gen + 1),
    torch.nn.Softmax()
).cpu()


# In[5]:


G_solver = []
for i in range(num_gen):
    G_solver.append(optim.Adam(G[i].parameters(), lr))
D_solver = optim.Adam(D.parameters(), lr)
###
loss = nn.CrossEntropyLoss()
label_G = Variable(torch.LongTensor(mb_size))
label_G = label_G.to(device)
label_D = Variable(torch.LongTensor(mb_size))
label_D = label_D.to(device)


# In[6]:


# Reset the gradients to zero
params = []
for i in range(num_gen):
    params.append(G[i])
params.append(D)
def reset_grad():
    for net in params:
        net.zero_grad()
reset_grad()


# In[7]:


data_index = 0
data = train_data
for it in range(1000):
    if ((data_index + 1)*mb_size>len(data)):
        data_index = 0

    data = torch.from_numpy(np.array(data[data_index*mb_size : (data_index + 1)*mb_size]))

    
    Total_D_loss = 0
    for i in range(num_gen):
        # Dicriminator forward-loss-backward-update
        #forward pass
        z = torch.FloatTensor(mb_size, Z_dim).uniform_(-1, 1)
        z = z.to(device)

        X = data[:, i*X_dim : (i+1)*X_dim]
        X = X.view(mb_size, X_dim)
        X = X.type(torch.FloatTensor)
        X = X.to(device)

        G_sample = G[i](z)
        D_real = D(X)
        D_fake = D(G_sample)
        # Calculate the loss
        D_loss_real = - torch.mean(D_real)#loss(D_real, label_D.fill_(num_gen + 0.1*randint(-1,1)))#TODO
        D_loss_fake = torch.mean(D_fake)#loss(D_fake, label_G.fill_(i + 0.1*randint(-1,1)))#TODO
        D_loss = D_loss_real + D_loss_fake
        # Total_D_loss = D_loss + Total_D_loss
        # Calulate and update gradients of discriminator
        D_solver.zero_grad()
        D_loss.backward()
        D_solver.step()

        # reset gradient
        # reset_grad()

    # Generator forward-loss-backward-update
    
    Total_G_loss = 0
    for i in range(num_gen):
        z = torch.FloatTensor(mb_size, Z_dim).uniform_(-1, 1)
        z = z.to(device)
        G_sample = G[i](z)
        D_fake = D(G_sample)
        G_loss = -torch.mean(D_fake)#loss(D_fake, label_D.fill_(num_gen + 0.1*randint(-1,1)))
        # Total_G_loss = G_loss + Total_G_loss
        G_solver[i].zero_grad()
        G_loss.backward()
        G_solver[i].step()

        commonDict = {k: v for k, v in G[i].state_dict().items() if 'former' in k}

        for j in range(num_gen):
            targetDict = G[j].state_dict()
            targetDict.update(commonDict)
            G[j].load_state_dict(targetDict)

        # reset gradient
        # reset_grad()

        # TODO: update common layer
        
    data_index = data_index + 1
    # Print and plot every now and then
    if it % 100 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'.format(it, D_loss.data.cpu().numpy(), G_loss.data.cpu().numpy()))


#  Let us see the images generated by the generator:

# In[18]:


import numpy as np
final = np.zeros((1500*mb_size,num_gen+1), dtype = float)
l = torch.zeros(mb_size, Z_dim)
for i in range(1500):
    z = torch.FloatTensor(mb_size, Z_dim).uniform_(-1, 1)
    z = z.to(device)
    for j in range(num_gen):
        l = torch.cat((l,G[j](z).cpu().detach()),1)
    final[i*mb_size : ((i + 1)*mb_size)] = transformer.inverse_transform(l.numpy(), None)
final = pd.DataFrame(final)
final.to_csv('out.csv')
# p1 = plt.hist(final, 500, density=True, histtype='bar', alpha=0.5)
# p2 = plt.hist(data, 500, density=True, histtype='bar', alpha=0.5)

