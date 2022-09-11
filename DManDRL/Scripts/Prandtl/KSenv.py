#! /home/linot/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:55:22 2020

@author: kevin
"""

import numpy as np
from CalcStats import Calculate
import matplotlib.pyplot as plt
import pickle

from scipy.integrate import odeint
import sys
sys.path.insert(0, '/home/floryan/odeNet/torchdiffeq')
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
               
        #Load Solutions
        #self.usol = np.expand_dims(np.loadtxt('u2.dat'),axis=1)
        #self.usolvect = np.loadtxt('u2.dat')
        
        #Load Stats Calculator
        self.GetStats = Calculate()
        
        #if trial!='':
        #    self.func=torch.load('/home/linot/Lorenz_NN/Lorenz_WorldModel/Unknown_Forcing/Train_KSE/'+trial+'/model.pt') # Trial1 is fullspace, Trial2 is PCA
        
        #self.trunc=12
        #[umean,U]=pickle.load(open('/home/linot/Lorenz_NN/Lorenz_WorldModel/Unknown_Forcing/Train_KSE/12_3/Data.p','rb'))
        #umean=np.mean(umean[:,:-4],axis=0)
        #self.PCA = U
        #self.umean = umean

        #Load Default Paths For ODENET, AE
        if auto_p=='':
            auto_p='/home/linot/KSNeuralNetwork/ODE/Larger/Vary_Dim/22_samp/Autoencoder/'+str(trunc)+'/Trial1'
            print('No AutoEncoder Path Given, Defaulting to backup path')
        if odenet_p=='':
            odenet_p='/home/linot/KS_Control/Unpertb_Auto/'+str(trunc)+'/Auto1/Trial2'
            print('No ODENet Path Given, Defaulting to backup path')


        sys.path.insert(0,'/home/kzeng/DeepRL/DDPG/KSE_Dreamz/Alec_Models')
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

    def step_dream(self,u0, action):

        # Integrate forward in time===========================
        ts = np.arange(0, self.dt*self.AC, self.dt)
        t=torch.tensor(ts)
        t=t.type(torch.FloatTensor)

        u0=torch.tensor(np.concatenate((u0[np.newaxis,np.newaxis,:],action[np.newaxis,np.newaxis,:]),axis=-1))
        u0=u0.type(torch.FloatTensor)
        uu =  torch_odeint(self.func, u0, t)
        uu=np.squeeze(uu.detach().numpy())

        self.u=uu[-1,:]
        #x = torch_odeint(self.func, self.state, ts) # CHANGE TO ODENET

        #Calculate Reward for Distance
        #reward = -1 * np.mean(np.linalg.norm(uu[:,:-4] - self.usol,axis=0))

        #Calculate Reward for diss============================

        #For PCA
        #Full_uu = uu[:,:-4]@self.PCA[:,:self.trunc].T + self.umean
        #Full_uu = Full_uu.T

        #For AE
        uu=np.asarray(uu)
        uu=uu.reshape((self.AC,-1))
        #print(uu.shape)
        Full_uu = self.decoder(uu[:,:-4])
        Full_uu = Full_uu.T
        #print(Full_uu.shape)

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
        #return self.usolvect

if __name__ == "__main__":
    
    ###########################################################################
    # Full Space
    ###########################################################################
    # Load environment
    env=KSenvironment(trial='Trial1')
    
    # Load eq
    dat=open('u1.dat','rb')
    usol=np.zeros(env.n)
    j=0
    for i in dat:
        usol[j]=float(i)
        j+=1
    env.usol=usol

    # Get IC
    u0=env.reset()

    uu=[]
    M=500
    for i in range(M):
        # Generate a random action
        act=2*(np.random.rand(4)-.5)

        # Evolve forward in the dream
        u0,_,_, uutemp=env.step_dream(u0,act)
        u0=u0[:-4]
        uu.append(uutemp)
        
    uu=np.asarray(uu)
    uu=uu.reshape((M*env.AC,-1))
    print(uu.shape)

    # Plot the trajectory
    plt.figure(figsize=(10,5))
    plt.pcolormesh(uu[:,:-4].transpose(),shading='gouraud')
    plt.savefig('Ex_Full.png')
    
    ###########################################################################
    # PCA Space
    ###########################################################################
    # Load environment
    env=KSenvironment(trial='12_3')
    
    # Load left singular vectors
    trunc=12
    [_,U]=pickle.load(open('/home/linot/Lorenz_NN/Lorenz_WorldModel/Unknown_Forcing/Train_KSE/Trial2/Data.p','rb'))
    
    # Load eq
    dat=open('u1.dat','rb')
    usol=np.zeros(env.n)
    j=0
    for i in dat:
        usol[j]=float(i)
        j+=1
    env.usol=usol@U[:,:trunc]

    # Get IC
    u0=env.reset()
    u0=u0@U[:,:trunc]

    uu=[]
    M=500
    for i in range(M):
        # Generate a random action
        act=2*(np.random.rand(4)-.5)

        # Evolve forward in the dream
        u0,_,_, uutemp=env.step_dream(u0,act)
        u0=u0[:-4]
        uu.append(uutemp)
        
    uu=np.asarray(uu)
    uu=uu.reshape((M*env.AC,-1))
    print(uu.shape)
    uuplot=uu[:,:-4]@U[:,:trunc].transpose()
    
    # Plot the trajectory
    plt.figure(figsize=(10,5))
    plt.pcolormesh(uuplot.transpose(),shading='gouraud')
    plt.savefig('Ex_PCA.png')


'''
import numpy as np
from CalcStats import Calculate

class KSenvironment:
    def __init__(self,L=16,N=128,dt=0.05,a_dim=4,AC=10,MaxRunTime=200,MaxAmp=1,FPenalty=1):
        self.L = L; 
        self.n = N; 
        self.dt = dt;
        
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
               
        #Load Solutions
        #self.usol = np.expand_dims(np.loadtxt('u2.dat'),axis=1)
        #self.usolvect = np.loadtxt('u2.dat')
        
        #Load Stats Calculator
        self.GetStats = Calculate()
        
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
            
            #Calculate Reward for Dissipation
            ux, uxx = self.GetStats.Calc(uu[0:64,:])
            Dissipation = np.mean(np.square(uxx),axis=0)
            MeanD = np.mean(Dissipation)
            
            #Calculate Reward for Power Input
            PowerInput = np.mean(np.square(ux),axis=0)
            MeanP = np.mean(PowerInput)
            
            #Calculate Reward for Forcing Input
            force = self.f0
            uf = np.mean(np.multiply(uu,np.expand_dims(force,axis=1)),axis=0)
            meanuf = np.mean(uf)            
            
            #Calculate Reward for Actuation Cost
            #ForceCost = np.sum(self.MaxAmp * action**2)
            #ForceCost = np.sum(self.MaxAmp * np.absolute(action))
            
            #Sum Rewards
            #reward = -1 * (MeanD + self.FPenalty * ForceCost) /1000
            reward = -1 * (MeanD + (MeanP + meanuf))/100
        
        else:
            reward = 0.0
            
        return self.u, reward, done, uu

    def reset(self):
        self.timer = 0.0
        no_action = np.zeros((self.a_dim,))
        u_old = np.random.uniform(-0.4,0.4,64)
        for i in range(800):
            u_new = self.advance(u_old,no_action)
            u_old = u_new
            
        return u_old
        #return self.usolvect

'''
