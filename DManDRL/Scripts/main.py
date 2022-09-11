#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:22:08 2022

@author: kevin
"""

#Basic Packages
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

#Tensorflow
import tensorflow
print(tensorflow.__version__)
from tensorflow import keras
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import json

#DDPG
from CalcStats import Calculate
from KSenv import KSenvironment
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
OU = OU()       #Ornstein-Uhlenbeck Process
GetStats = Calculate() #Calc 1st and 2nd spatial derivatives

#PyTorch
from scipy.integrate import odeint
import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim

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

#Load Parameters from Parameters.txt
def LoadParameters(filename):
    p_file = open(filename,'r')
    inputs = []
    for line in p_file:
            columns=line.split()
            text=''
            space=''
            for i in columns:
                if i[0]!='#':
                    text=text+space+i
                    space=' '
                else:
                    break
            inputs.append(text)
    return inputs

#DDPG Training Algorithm
def TrainDDPG(BUFFER_SIZE,BATCH_SIZE,GAMMA,TAU,LRA,LRC,EXPLORE,episode_count,max_steps,AC,dt,MaxAmp,MaxRunTime,Fpenalty,STATE_DIM,ACTION_DIM,trunc,auto_p,odenet_p):
    train_indicator=1    #1 for Train, 0 for Run
    #np.random.seed(1337)

    plot_r = []            #Store Rewards
    done = False            #Initialize Done State
    step = 0                #Initialize step counter
    epsilon = 1             #Initialize Exploration Factor

    #Tensorflow GPU optimization (MAY NOT BE NEEDED)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from tensorflow.keras import backend as K
    K.set_session(sess)

    #Build actor and critic networks, initialize buffer
    actor = ActorNetwork(sess, STATE_DIM, ACTION_DIM, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, STATE_DIM, ACTION_DIM, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)

    #Initialize Kuramoto-Sivashinsky Environment
    env = KSenvironment(trunc=trunc,auto_p=auto_p,odenet_p=odenet_p)
    print("Kuramoto-Sivashinsky Environment has been initialized")
    
    #Episode Training Loop
    for i in range(episode_count):
        #print("Episode : " + str(i) +"/" + str(episode_count) + " Replay Buffer " + str(buff.count()) + " Epsilon : " + str(epsilon))
        
        #Reset Reward, Initial System State (Randomly on Attractor)
        ob_t_full = env.reset()

        #For AE
        ob_t = env.encoder(ob_t_full)
        ob_t = np.squeeze(ob_t)

        s_t = ob_t #Change this for limited observation
        total_reward = 0.0
        
        #Training a single trajectory
        for j in range(max_steps):
            loss = 0                            #Reset Loss to 0
            epsilon -= 1.0 / EXPLORE            #Decrease Exploration
            epsilon = max(epsilon, 0.1)
            a_t = np.zeros([1,ACTION_DIM])      #Initialize Action Vector
            noise_t = np.zeros([1,ACTION_DIM])  #Initialize Noise Vector
            
            #Agent Produces Action given the state with Random Exploration (Invariant)
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))

# =============================================================================
#             #Diminishing Noise
#             noise_t = max(epsilon, 0.01) * OU.function(a_t_original[0], 0.0, 0.30, 0.20) # OU(x, mu, theta, sigma)
#             a_t = a_t_original + noise_t
#             a_t = np.clip(a_t, -1.0, 1.0)
# =============================================================================
            
            #Greedy Noise Selection
            if np.random.rand() <= epsilon:
                noise_t = max(epsilon, 1.0) * OU.function(a_t_original[0], 0.0, 0.30, 0.20) # OU(x, mu, theta, sigma)
                a_t = a_t_original + noise_t
                a_t = np.clip(a_t, -1, 1)
            else:
                a_t[0] = a_t_original

            ob_t1, r_t, done, _ = env.step_dream(ob_t,np.squeeze(a_t[0]))
            s_t1 = ob_t1[:-4]
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Store trajectory to memory cache, these are all invariant quantities

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)                   #Sample from memory cache
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])     #Calculate Q value
           
            for k in range(len(batch)): #Calculate Bellman RHS
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator): #Fit Actor and Critic Networks 
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()
            
            #Update Reward, set new state to current state, update step counter, check for stop
            total_reward += r_t     #Sum rewards
            s_t = s_t1              #Update Invariant State
            ob_t = ob_t1[:-4]            #Update True State
            
            step += 1
            if done:
                break
        
        #Save Model every 5 Episodes
        if np.mod(i, 5) == 0:
            if (train_indicator):
                #print("Saving Model...")
                actor.model.save_weights("../Models/RLagents/dream_KSE_AE12_actormodel.h5", overwrite=True)
                with open("../Models/RLagents/dream_KSE_AE12_actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("../Models/RLagents/dream_KSE_AE12_criticmodel.h5", overwrite=True)
                with open("../Models/RLagents/dream_KSE_AE12_criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)
        
        #Print Model Performance
        #print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        plot_r.append(total_reward)
        #print("Total Step: " + str(step))
        #print("")
        
    #Plot and Save Training Results
    pickle.dump(plot_r, open( "data_r.p", "wb" ))

    fig = plt.figure()
    plt.plot(plot_r,lw=1.0,c='k')
    plt.xlabel("Episodes")
    plt.ylabel("Training Reward")
    fig.set_size_inches(5, 5)
    plt.savefig("F_Training.png", bbox_inches = 'tight', pad_inches = 0.4)

    #from tensorflow.keras import backend as K
    #K.clear_session()

def Test_Agent(AC,dt,MaxAmp,MaxRunTime,Fpenalty,STATE_DIM,ACTION_DIM,trunc,auto_p,odenet_p,run_num):
    BATCH_SIZE = 32      #Sample Size of Gradient
    TAU = 0.01     #Target Network HyperParameters 0.001
    LRA = 0.001    #Learning rate for Actor 0.0001
    LRC = 0.01     #Learning rate for Critic 0.001

    #np.random.seed(1337)

    EXPLORE = 20.      #Decay Factor 1/EXPLORE
    episode_count = 1   #Number Episodes
    max_steps = 1000     #Episode Step Limit
    done = False           #Initialize Done State
    step = 0               #Initialize step counter
    epsilon = 0            #Initialize Exploration Factor
    testnum = run_num       

    #Tensorflow GPU optimization (MAY NOT BE NEEDED)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from tensorflow.keras import backend as K
    K.set_session(sess)
    K.set_floatx('float32')

    #Build Networks
    actor = ActorNetwork(sess, STATE_DIM, ACTION_DIM, BATCH_SIZE, TAU, LRA)
    #critic = CriticNetwork(sess, STATE_DIM, ACTION_DIM, BATCH_SIZE, TAU, LRC)

    #Now load the weight
    print("Loading weights")
    try:        #Attempts to load, if fails goes to next
        actor.model.load_weights("../Models/RLagents/dream_KSE_AE12_actormodel.h5")
        actor.target_model.load_weights("../Models/RLagents/dream_KSE_AE12_actormodel.h5")
        print("Weights load successfully")
    except:
        print("Cannot find the weight")

    # Initialize Environment
    #For AE
    env = KSenvironment(trunc=trunc,auto_p=auto_p,odenet_p=odenet_p)


    for i in range(episode_count):
        ob_t_full = env.reset()
        ob_t_full = pickle.load(open("../Data/IC"+str(testnum)+".p", "rb" ))
        #For AE
        ob_t = env.encoder(ob_t_full)
        ob_t = np.squeeze(ob_t)
    
        s_t = ob_t #Change this for limited observation
        total_reward = 0.0
        
        #Initialize Storage Vectors
        a_log = np.zeros((ACTION_DIM,1))
        u_log = np.expand_dims(ob_t_full,axis=1)
        
        for j in range(max_steps):
            epsilon -= 1.0 / EXPLORE            #Decrease Exploration
            a_t = np.zeros([1,ACTION_DIM])      #Initialize Action Vector
            a_t[0] = actor.model.predict(s_t.reshape(1, s_t.shape[0]))    #Get first 
            
            #Implement Action in Environment
            ob_t1_full, done, u_step = env.step(ob_t_full,a_t[0])
            s_t1_full = ob_t1_full
            
            #Store Data for plotting
            u_log = np.concatenate((u_log, u_step[:,1:]), axis=1)
            a_step = np.tile(a_t[0],(AC+1,1)).T
            a_log = np.concatenate((a_log, a_step[:,1:]), axis=1)
            
            #Update State
            #total_reward += r_t     #Sum rewards
            s_t_full = s_t1_full              #Update State
            ob_t_full = ob_t1_full

            #For AE
            s_t = env.encoder(s_t_full)
            s_t = np.squeeze(s_t)

            step += 1
            if done:
                print("Time Limit Reached.")
                break

        #print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        #print("Total Step: " + str(step))
        #print("")

    print("Simulation Complete.")
    
    #Save u_log and a_log Data
    pickle.dump(u_log, open( "data_u"+str(testnum)+".p", "wb" ))
    pickle.dump(a_log, open( "data_a"+str(testnum)+".p", "wb" ))

    #Plot Trajectory Result----------------------------------------------------
    colormap = sns.diverging_palette(240, 10, as_cmap=True)
    N = 64
    L = 22
    x_grid = np.arange(N)*L/N
    t_log = np.linspace(0,max_steps*dt*AC,max_steps*AC+1)
    fig = plt.figure()
    plt.pcolormesh(t_log,x_grid,u_log,cmap=colormap,vmin=-3, vmax=3,shading='gouraud')
    fig.set_size_inches(10, 2)
    plt.savefig("F_Trajectory"+str(testnum)+".png", bbox_inches = 'tight', pad_inches = 0.4)
    plt.close()
        
    #Compute force field----------------------------------------------------------
    B = env.B
    f0 = np.zeros((x_grid.size,1))
    dum = np.zeros((x_grid.size,ACTION_DIM))
    
    forcefield = np.zeros((x_grid.size,max_steps*AC+1))    
    for j in range (0,max_steps*AC+1):
        for i in range(0,ACTION_DIM):
            dum[:,i] = B[:,i]*a_log[i,j]
        f0 = np.sum(dum, axis=1)
        forcefield[:,j] = f0

    #Calculate Spatial Statistics----------------------------------------------     
    ux, uxx = GetStats.Calc(u_log[0:64,:])
    Dissipation = np.mean(np.square(uxx),axis=0)
    RawPower = np.mean(np.square(ux),axis=0)
    uf = np.mean(np.multiply(forcefield,u_log),axis=0)
    TotalPower = RawPower + uf
    
    #Plot Performance----------------------------------------------------------
    fig = plt.figure()
    plt.plot(t_log,Dissipation,c='b',alpha=0.7,label='D')
    plt.plot(t_log,TotalPower,c='r',alpha=0.7,label='Pf')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim(0,2.5)
    axes.set_xlim(0,max_steps*dt*AC)
    fig.set_size_inches(10, 2)
    plt.savefig("F_StatTrajectory"+str(testnum)+".png", bbox_inches = 'tight', pad_inches = 0.4)
    plt.close()

    print('True System Test Complete')
    
    
def Test_DManD_Agent(AC,dt,MaxAmp,MaxRunTime,Fpenalty,STATE_DIM,ACTION_DIM,trunc,auto_p,odenet_p,run_num):
    BATCH_SIZE = 32      #Sample Size of Gradient
    TAU = 0.01     #Target Network HyperParameters 0.001
    LRA = 0.001    #Learning rate for Actor 0.0001
    LRC = 0.01     #Learning rate for Critic 0.001

    #np.random.seed(1337)

    EXPLORE = 20.      #Decay Factor 1/EXPLORE
    episode_count = 1   #Number Episodes
    max_steps = 1000     #Episode Step Limit 1000
    done = False           #Initialize Done State
    step = 0               #Initialize step counter
    epsilon = 0            #Initialize Exploration Factor

    testnum = run_num         

    #Tensorflow GPU optimization (MAY NOT BE NEEDED)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from tensorflow.keras import backend as K
    K.set_session(sess)
    K.set_floatx('float32')

    #Build Networks
    actor = ActorNetwork(sess, STATE_DIM, ACTION_DIM, BATCH_SIZE, TAU, LRA)
    #critic = CriticNetwork(sess, STATE_DIM, ACTION_DIM, BATCH_SIZE, TAU, LRC)

    #For AE
    env = KSenvironment(trunc=trunc,auto_p=auto_p,odenet_p=odenet_p)

    #Now load the weight
    print("Loading weights")
    try:        #Attempts to load, if fails goes to next
        actor.model.load_weights("../Models/RLagents/dream_KSE_AE12_actormodel.h5")
        actor.target_model.load_weights("../Models/RLagents/dream_KSE_AE12_actormodel.h5")
        print("Weights load successfully")
    except:
        print("Cannot find the weight")

    for i in range(episode_count):
        ob_t_full = env.reset()
        ob_t_full = pickle.load(open("../Data/IC"+str(testnum)+".p", "rb" ))
        #For AE
        ob_t = env.encoder(ob_t_full)
        ob_t = np.squeeze(ob_t)
    
        s_t = ob_t #Change this for limited observation
        total_reward = 0.0
        
        #Initialize Storage Vectors
        a_log = np.zeros((ACTION_DIM,1))
        h_log = np.expand_dims(ob_t,axis=1)
        
        for j in range(max_steps):
            epsilon -= 1.0 / EXPLORE            #Decrease Exploration
            a_t = np.zeros([1,ACTION_DIM])      #Initialize Action Vector
            a_t[0] = actor.model.predict(s_t.reshape(1, s_t.shape[0]))    #Get first 
            
            #Implement Action in Environment
            ob_t1, _, done, h_step = env.step_DManD(ob_t,np.squeeze(a_t[0]))
            s_t1 = ob_t1[:-4]

            #Store Data for plotting
            h_log = np.concatenate((h_log, h_step[:,:-4].T), axis=1)
            a_step = np.tile(a_t[0],(AC+1,1)).T
            a_log = np.concatenate((a_log, a_step[:,1:]), axis=1)
            
            #Update State
            #total_reward += r_t     #Sum rewards
            s_t = s_t1              #Update State
            ob_t = ob_t1[:-4]

            step += 1
            if done:
                print("Time Limit Reached.")
                break

        #print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        #print("Total Step: " + str(step))
        #print("")

    print("Simulation Complete.")
    
    #Save u_log and a_log Data
    pickle.dump(h_log, open( "data_NODE_h"+str(testnum)+".p", "wb" ))
    pickle.dump(a_log, open( "data_NODE_a"+str(testnum)+".p", "wb" ))

    #Compute Reconstruction h back to u

    #For AE
    u_log=env.decoder(h_log.T)
    u_log = u_log.T

    #For PCA
    pickle.dump(u_log, open( "data_NODE_u"+str(testnum)+".p", "wb" ))

    #Plot Trajectory Result----------------------------------------------------
    colormap = sns.diverging_palette(240, 10, as_cmap=True)
    N = 64
    L = 22
    x_grid = np.arange(N)*L/N
    t_log = np.linspace(0,max_steps*dt*AC,max_steps*AC+1)
    fig = plt.figure()
    plt.pcolormesh(t_log,x_grid,u_log,cmap=colormap,vmin=-3, vmax=3,shading='gouraud')
    fig.set_size_inches(10, 2)
    plt.savefig("Fig_NODE_Trajectory"+str(testnum)+".png", bbox_inches = 'tight', pad_inches = 0.4)
    plt.close()
        
    #Compute force field----------------------------------------------------------
    B = env.B
    f0 = np.zeros((x_grid.size,1))
    dum = np.zeros((x_grid.size,ACTION_DIM))
    
    forcefield = np.zeros((x_grid.size,max_steps*AC+1))    
    for j in range (0,max_steps*AC+1):
        for i in range(0,ACTION_DIM):
            dum[:,i] = B[:,i]*a_log[i,j]
        f0 = np.sum(dum, axis=1)
        forcefield[:,j] = f0

    #Calculate Spatial Statistics----------------------------------------------        
    ux, uxx = GetStats.Calc(u_log[0:64,:])
    Dissipation = np.mean(np.square(uxx),axis=0)
    RawPower = np.mean(np.square(ux),axis=0)
    uf = np.mean(np.multiply(forcefield,u_log),axis=0)
    TotalPower = RawPower + uf
    
    #Plot Performance----------------------------------------------------------
    fig = plt.figure()
    plt.plot(t_log,Dissipation,c='b',alpha=0.7,label='D')
    plt.plot(t_log,TotalPower,c='r',alpha=0.7,label='Pf')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim(0,2.5)
    axes.set_xlim(0,max_steps*dt*AC)
    fig.set_size_inches(10, 2)
    plt.savefig("Fig_NODE_StatTrajectory"+str(testnum)+".png", bbox_inches = 'tight', pad_inches = 0.4)
    plt.close()
    
    
if __name__ == "__main__":
    # Read parameters of training
    Para = LoadParameters('Parameters.txt')
    BUFFER_SIZE = int(Para[0])      # Replay Buffer Size
    BATCH_SIZE = int(Para[1])       # Optimizer Batch Size
    
    GAMMA = float(Para[2])          # Reward Discount Factor
    TAU = float(Para[3])            # Master Network Update Rate
    LRA = float(Para[4])            # Learning Rate Actor Network
    LRC = float(Para[5])            # Learning Rate Critic Network
    
    EXPLORE = float(Para[6])        # Exploration Value
    episode_count = int(Para[7])    # Episode Count
    max_steps = int(Para[8])        # Max Steps 
    
    AC = int(Para[9])               # Action Commitment (time steps)
    dt = float(Para[10])            # Integral Time Steps
    
    MaxAmp = float(Para[11])        # Maximum forcing coefficient
    MaxRunTime = float(Para[12])    # Maximum Run Time
    Fpenalty = float(Para[13])      # Forcing Penalty
    STATE_DIM = int(Para[14])       # Dimension of State Observation
    ACTION_DIM = int(Para[15])      # Dimension of Action

    trunc = 12                      # Truncation Value

    #Construct Paths
    odenet_p='../Models/node'
    auto_p='../Models/autoencoder'
    
    # Train DDPG Network
    #TrainDDPG(BUFFER_SIZE,BATCH_SIZE,GAMMA,TAU,LRA,LRC,EXPLORE,episode_count,max_steps,AC,dt,MaxAmp,MaxRunTime,Fpenalty,STATE_DIM,ACTION_DIM,trunc,auto_p,odenet_p)
    #print("Training Complete.")
    
    # Test DDPG Network
    Test_Agent(AC,dt,MaxAmp,MaxRunTime,Fpenalty,STATE_DIM,ACTION_DIM,trunc,auto_p,odenet_p,1)

    # Test DManD Network
    Test_DManD_Agent(AC,dt,MaxAmp,MaxRunTime,Fpenalty,STATE_DIM,ACTION_DIM,trunc,auto_p,odenet_p,1)
