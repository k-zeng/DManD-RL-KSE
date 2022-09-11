#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:25:51 2022

@author: kevin
"""

import numpy as np
from CalcStats import Calculate
import matplotlib.pyplot as plt
import pickle

from scipy.integrate import odeint
import sys
#sys.path.insert(0, '/home/floryan/odeNet/torchdiffeq')
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint as torch_odeint

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
    
# =============================================================================
#     def forward_scipy(self, t, y):
#         [M,K,N]=y.shape
#         y_NN=self.net(y)
#         y_NN=torch.cat((y_NN,torch.zeros(M,K, 4, dtype=y_NN.dtype, device=y_NN.device)), dim=-1)
# 
#         return y_NN
# =============================================================================

class KSenvironment:
    def __init__(self,L=22,N=64,dt=0.05,a_dim=4,AC=5,MaxRunTime=250,MaxAmp=1,FPenalty=1,trunc=None,auto_p='',odenet_p=''):
        self.L = L 
        self.n = N
        self.dt = dt
        
        #Define New Parameters
        self.AC = AC
        self.timer = None
        self.runtime = MaxRunTime
        self.MaxAmp = MaxAmp
        self.FPenalty = FPenalty
        
        #Define Wavenumbers and linear component of KSE
        self.x = np.arange(N)*L/N
        self.k = N*np.fft.fftfreq(N)[0:N//2+1]*2*np.pi/L
        self.ik    = 1j*self.k                   # spectral derivative operator
        self.lin   = self.k**2 - self.k**4       # Fourier multipliers for linear term
        
        #Define Gaussian Jets
        self.a_dim = a_dim
        self.B = np.zeros((self.x.size,self.a_dim))
        sig = 0.4
        x_zero = self.x[-1]/a_dim*np.arange(0,a_dim)
        gaus = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*((self.x-self.x[self.x.size//2])/sig)**2)
        for i in range(0,a_dim):
            #self.B[:,i] = np.roll(gaus,int(np.floor(x_zero[i]-self.x[self.x.size//2])/(self.x[1]-self.x[0])))
            self.B[:,i] = np.roll(gaus, i*16)
            self.B[:,i] = 0.50 * self.B[:,i]/max(self.B[:,i])
            self.B[:,i] = np.roll(self.B[:,i],-24)
               
        
        #Load Stats Calculator
        self.GetStats = Calculate()
        
        #Load Default Paths For ODENET, AE
        if auto_p=='':
            auto_p='../Models/autoencoder'
            print('No AutoEncoder Path Given, Defaulting to backup path')
        if odenet_p=='':
            odenet_p='../Models/node'
            print('No ODENet Path Given, Defaulting to backup path')

        import buildmodel


        #Load normalization info
        [u_mean,u_std,U]=pickle.load(open(auto_p+'/PCA.p','rb'))
        self.u_mean=u_mean
        self.u_std=u_std
        self.U=U
        #Load encoder and decoder
        model,_,_=buildmodel.buildmodel(N)
        model.load_weights(auto_p+'/model.h5')
        encode=buildmodel.encode(N)
        decode=buildmodel.decode(N)
        for i in range(2):
            ly='Dense_In'+str(i+1)
            encode.get_layer(ly).set_weights(model.get_layer(ly).get_weights())
            ly='Dense_Out'+str(i+1)
            decode.get_layer(ly).set_weights(model.get_layer(ly).get_weights())
        self.encode=encode
        self.decode=decode

        #Load odenet
        self.func=torch.load(odenet_p+'/model.pt')
    
    def encoder(self,u):
        
        utemp=(u-self.u_mean[np.newaxis,:])/self.u_std[np.newaxis,:]
        a=utemp@ self.U

        return self.encode.predict(a)

    def decoder(self,h):
        
        a=self.decode.predict(h)
        utemp=a@self.U.transpose()
        u=utemp*self.u_std[np.newaxis,:]+self.u_mean[np.newaxis,:]

        return u


    def nlterm(self,u,f):
        # compute tendency from nonlinear term. advection + forcing
        ur = np.fft.irfft(u,axis=-1)
        return -0.5*self.ik*np.fft.rfft(ur**2,axis=-1)+f
    
    def advance(self,u0,action):
        #forcing shape
        self.f0 = np.zeros((self.x.size,1))
        dum = np.zeros((self.x.size,self.a_dim))
        
        for i in range(0,action.size):
            dum[:,i] = self.B[:,i]*action[i]
        self.f0 = np.sum(dum, axis=1)
        
        # semi-implicit third-order runge kutta update.
        # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
        self.u = np.fft.rfft(u0,axis=-1)
        self.f = np.fft.rfft(self.f0,axis=-1)
        u_save = self.u.copy()
        for n in range(3):
            dt = self.dt/(3-n)
            # explicit RK3 step for nonlinear term
            self.u = u_save + dt*self.nlterm(self.u,self.f)
            # implicit trapezoidal adjustment for linear term
            self.u = (self.u+0.5*self.lin*dt*u_save)/(1.-0.5*self.lin*dt)
        
        self.u = np.fft.irfft(self.u,axis=-1)
        return self.u
    
    def step(self,u0,action):
        #initialize storage array with starting u
        uu = np.expand_dims(u0,axis=1)
        u_old = u0
        for i in range(self.AC):
            u_new = self.advance(u_old,action)
            uu = np.concatenate((uu,np.expand_dims(u_new,axis=1)),axis=1)
            u_old = u_new
            self.timer += self.dt
            
        #Check stop condition
        done = self.timer > self.runtime
        done = bool(done)
        if not done:
            #Calculate Reward for Distance
            #L2Distance = np.linalg.norm(uu - self.usol,axis=0)
            #MeanL2 = np.mean(L2Distance)
            
            reward = 0.0
        
        else:
            reward = 0.0
            
        #return self.u, reward, done, uu
        return self.u, done, uu

    def step_DManD(self,u0, action):

        # Integrate forward in time===========================
        ts = np.arange(0, self.dt*self.AC, self.dt)
        t=torch.tensor(ts)
        t=t.type(torch.FloatTensor)

        u0=torch.tensor(np.concatenate((u0[np.newaxis,np.newaxis,:],action[np.newaxis,np.newaxis,:]),axis=-1))
        u0=u0.type(torch.FloatTensor)
        uu =  torch_odeint(self.func, u0, t)
        #uu =  odeint(self.func.forward_scipy, np.squeeze(u0), t)
        uu=np.squeeze(uu.detach().numpy())

        self.u=uu[-1,:]

        #For AE
        uu=np.asarray(uu)
        uu=uu.reshape((self.AC,-1))
        Full_uu = self.decoder(uu[:,:-4])
        Full_uu = Full_uu.T

        #Calculate Reward for Dissipation=======================
        ux, uxx = self.GetStats.Calc(Full_uu[0:64,:])
        Dissipation = np.mean(np.square(uxx),axis=0)
        MeanD = np.mean(Dissipation)
            
        #Calculate Reward for Power Input
        PowerInput = np.mean(np.square(ux),axis=0)
        MeanP = np.mean(PowerInput)
            
        #Calculate Reward for Forcing Input
        force = self.f0
        uf = np.mean(np.multiply(Full_uu,np.expand_dims(force,axis=1)),axis=0)
        meanuf = np.mean(uf)            
            
        #Sum Rewards============================================
        reward = -1 * (MeanD + (MeanP + meanuf))/100

        return self.u, reward, False, uu

    def reset(self):
        self.timer = 0.0
        no_action = np.zeros((self.a_dim,))
        u_old = np.random.uniform(-0.4,0.4,64)
        for i in range(800):
            u_new = self.advance(u_old,no_action)
            u_old = u_new
            
        return u_old

if __name__ == "__main__":
    
    ###########################################################################
    # Full Space
    ###########################################################################
    # Load environment
    print('KSE RL Env with NODE Env')

