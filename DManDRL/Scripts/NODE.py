#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:55:35 2019

@author: K-Dawg + Alec (Pretty much just Alec at this point)
"""

import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.utils.extmath import randomized_svd

import sys
#sys.path.insert(0, '/home/floryan/odeNet/torchdiffeq')
#sys.path.insert(0, '/home/kzeng/DeepRL/DDPG/KSE_Dreamz/Alec_Models') # Change to buildmodel directory
import buildmodel
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle

###############################################################################
# Arguments from the submit file
###############################################################################
parser = argparse.ArgumentParser('ODE demo')
# These are the relevant sampling parameters
parser.add_argument('--data_size', type=int, default=100)  #IC from the simulation
parser.add_argument('--batch_time', type=int, default=1)   #Samples a batch covers (this is 2 snaps in a row in data_size)
parser.add_argument('--batch_size', type=int, default=20)   #Number of IC to calc gradient with each iteration

parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--niters', type=int, default=100)       #Iterations of training
parser.add_argument('--test_freq', type=int, default=20)    #Frequency for outputting test loss
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

#Add 1 to batch_time to include the IC
args.batch_time+=1 

# Determines what solver to use
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    # This is the default
    from torchdiffeq import odeint

# Check if there are gpus
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

###############################################################################
# Classes
###############################################################################

# This is the class that contains the NN that estimates the RHS
class ODEFunc(nn.Module):
    def __init__(self,N1,N2):
        super(ODEFunc, self).__init__()
        # Change the NN architecture here
        self.net = nn.Sequential(
            nn.Linear(N1, 200),
            nn.Sigmoid(),
            nn.Linear(200, 200),
            nn.Sigmoid(),
            nn.Linear(200,200),
            nn.Sigmoid(),
            nn.Linear(200, N2),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        [M,K,N]=y.detach().numpy().shape
        y_NN=self.net(y)
        y_NN=torch.cat((y_NN,torch.zeros(M,K, 4, dtype=y_NN.dtype, device=y_NN.device)), dim=-1)

        return y_NN

# This class is used for updating the gradient
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

###############################################################################
# Functions
###############################################################################
# Gets a batch of y from the data evolved forward in time (default 20)
def get_batch(t,true_y,reset,M_train):
    vals=np.arange(M_train - args.batch_time, dtype=np.int64)
    vals=np.delete(vals,reset)
    s = torch.from_numpy(np.random.choice(vals, args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

# For outputting info when running on compute nodes
def output(text):
    # Output epoch and Loss
    name='Out.txt'
    newfile=open(name,'a+')
    newfile.write(text)
    newfile.close()

if __name__ == "__main__":
    
    ###########################################################################
    # Load data
    ###########################################################################
    M=40000
    N=64
    N_act=4
    path=os.path.abspath(os.getcwd())
    auto=path.split('/')[9] # Change to get the correct autoencoder
    trunc=buildmodel.trunc()
    [u_mean,u_std,U]=pickle.load(open('../Models/autoencoder/PCA.p','rb'))
    [u,reset]=pickle.load(open('../Data/Data.p','rb'))

    frac=.8
    tempu=np.copy(u[:,:N])
    u_train=tempu[:round(M*frac),:]
    act_train=u[:round(M*frac),N:]
    u_test=tempu[round(M*frac):M,:]

    u_train=(u_train-u_mean[np.newaxis,:])/u_std[np.newaxis,:]
    u_test=(u_test-u_mean[np.newaxis,:])/u_std[np.newaxis,:]

    # Do a full change of basis
    a_train=u_train@ U
    a_test=u_test@ U

    ###########################################################################
    # Encode data
    ###########################################################################

    # Load encoder
    model_path='../Models/autoencoder/model.h5'
    model,_,_=buildmodel.buildmodel(N)
    model.load_weights(model_path)
    encode=buildmodel.encode(N)
    for i in range(2):
        ly='Dense_In'+str(i+1)
        encode.get_layer(ly).set_weights(model.get_layer(ly).get_weights())

    htemp=encode.predict(a_train)

    h_train=np.zeros((round(M*frac),trunc+N_act))
    h_train[:,:trunc]=htemp
    h_train[:,trunc:]=act_train

    # Train ODENet with these state action pairs
    true_y=torch.tensor(h_train[:,np.newaxis,:])
    true_y=true_y.type(torch.FloatTensor)
    ts=np.arange(0,M*.25,.25)
    ts=ts[:round(M*frac)]
    t=torch.tensor(ts)
    t=t.type(torch.FloatTensor)

    ###########################################################################
    # Initialize NN for learning the RHS and setup optimization parms
    ###########################################################################
    func = ODEFunc(trunc+N_act,trunc)
    optimizer = optim.Adam(func.parameters(), lr=1e-3) #optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    err=[]
    ii = 0
    ###########################################################################
    # Optimization iterations
    ###########################################################################
    for itr in range(1, args.niters + 1):

        # Get the batch and initialzie the optimizer
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(t,true_y,reset,round(M*frac))
        if itr==1:
            output('Batch Time Units: '+str(batch_t.detach().numpy()[-1])+'\n')
            #print('Batch Time Units: '+str(batch_t.detach().numpy()[-1]))

        # Make a prediction and calculate the loss
        pred_y = odeint(func, batch_y0, batch_t)
        loss = torch.mean(torch.abs(pred_y - batch_y)) # Compute the mean (because this includes the IC it is not as high as it should be)
        loss.backward() #Computes the gradient of the loss (w.r.t to the parameters of the network?)
        # Use the optimizer to update the model
        optimizer.step()

        # Print out the Loss and the time the computation took
        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        if itr % args.test_freq == 0:
            with torch.no_grad():
                err.append(loss.item())
                output('Iter {:04d} | Total Loss {:.6f} | Time {:.6f}'.format(itr, loss.item(),time.time() - end)+'\n')
                #print('Iter {:04d} | Total Loss {:.6f} | Time {:.6f}'.format(itr, loss.item(),time.time() - end))
                ii += 1
        end = time.time()
    
    ###########################################################################
    # Plot results and save the model
    ###########################################################################
    torch.save(func, 'model.pt')
    #pickle.dump(func,open('model.p','wb'))

    # Plot the learning
    plt.figure()
    plt.plot(np.arange(args.test_freq,args.niters+1,args.test_freq),np.asarray(err),'.-')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.savefig('Error_v_Epochs.png')
