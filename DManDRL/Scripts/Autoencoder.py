#!/usr/bin/env python3
"""
Created on Wed Aug 14 2019

Dense invariant NN with tied encoder and decoder

@author: Alec
"""

import os
import sys
import math

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import scipy.io
from sklearn.utils.extmath import randomized_svd
#sys.path.insert(0, '/home/kzeng/DeepRL/DDPG/KSE_Dreamz/Alec_Models') # put buildmodel directory here
import buildmodel

#sys.path.insert(0,'/home/linot/Lorenz_NN/Lorenz_WorldModel')
from KSEjet_generate import KSenvironment


# Class for printing history
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
      newfile=open(name,'a+')
      newfile.write('Epoch: '+str(epoch)+'   ')
      newfile.write('Loss: '+str(logs['loss'])+'\n')
      newfile.close()

###############################################################################
# Plotting Functions
###############################################################################
# For plotting history
def plot_history(histshift):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Dense Shift Loss')
    plt.semilogy(histshift['epoch'], histshift['loss'],
       label='Train Loss')
    plt.semilogy(histshift['epoch'], histshift['val_loss'],
       label = 'Val Loss')
    plt.legend()
    #plt.ylim([10**-6,10**-1])


def plot_parity(utt_test,test_predictions,KSmax=3.5):
    plt.plot(utt_test.flatten(), test_predictions.flatten(),'o',color='black',markersize=.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([-KSmax,KSmax])
    plt.ylim([-KSmax,KSmax])
    plt.plot([-100, 100], [-100, 100])

def plot_hist(error,title,bins=40,ymax=50,xmax=.05):
    plt.hist(error, bins = bins,density=True)
    plt.xlim([-xmax,xmax])
    plt.ylim([0,ymax])
    plt.xlabel("Prediction Error")
    variance=np.mean(error**2)
    plt.title('MSE='+str(round(variance,7)))
    plt.ylabel(title+' PDF')
    
    return variance

def scheduler(epoch, lr):
    if epoch < 500:
        return .001
    else:
        return .0001

###############################################################################
# Main Function
###############################################################################
if __name__=='__main__':
    ###########################################################################
    # Generate Data
    ###########################################################################
    # Import Lorenz environment
    env = KSenvironment() 
    # Reset to a get a reasonable IC
    u0=env.reset()

    # Generate possible force combinations
    # Evolve forward many different IC+action pairs
    M=40000 # Can vary
    N=64
    N_act=4
    u=np.zeros((M,N+N_act))
    u[0,:N]=env.u
    reset_time=200 # This varies how frequently the state is reset
    reset=[]
    for i in range(M-1):
        if i%reset_time==0 and i!=0:
            reset.append(i)
            # Reset the state (Don't set action because it is unused)
            u[i+1,:N]=env.reset()
        else:
            # Pick a random forcing
            u[i,N:]=2*(np.random.rand(4)-.5)
            u[i+1,:N],_,_=env.step(u[i,:N],u[i,N:])

    pickle.dump([u,reset],open('Data.p','wb'))

    # Remove the last 20% of the data for testing on
    frac=.8
    tempu=np.copy(u[:,:N])
    u_train=tempu[:round(M*frac),:]
    u_test=tempu[round(M*frac):M,:]

    # Output sizes for validation
    name='Out.txt'
    newfile=open(name,'a+')
    newfile.write('Training Shapes\n')
    newfile.write(str(u_train.shape)+'\n')
    newfile.write('Testing Shapes\n')
    newfile.write(str(u_test.shape)+'\n')
    newfile.close()

    ###########################################################################
    # PCA Encoder
    ###########################################################################
    u_mean=np.mean(u_train,axis=0)
    u_std=np.std(u_train,axis=0)
    u_train=(u_train-u_mean[np.newaxis,:])/u_std[np.newaxis,:]
    u_test=(u_test-u_mean[np.newaxis,:])/u_std[np.newaxis,:]

    # Perform SVD on the training data
    U,S,VT=randomized_svd(u_train.transpose(), n_components=N)
    # Do a full change of basis
    a_train=u_train@ U
    a_test=u_test@ U

    pickle.dump([u_mean,u_std,U],open('PCA.p','wb'))

    ###########################################################################
    # Build autoencoder
    ###########################################################################
    # Build a new model with a timestepping layer (built here so I know how many PCA modes to keep)
    model,EPOCHS,optimizer=buildmodel.buildmodel(N)

    # Compile model
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae', 'mse'])

    print(model.summary())
    
    # Train the model 
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(a_train, a_train,epochs=EPOCHS, validation_data=(a_test[:1000:10,:],a_test[:1000:10,:]), verbose=0,callbacks=[PrintDot(),callback])

    # Save history for comparing models
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    ###########################################################################
    # Plot Results
    ###########################################################################
    
    # Plot MAE and MSE vs epochs
    plot_history(hist)
    plt.tight_layout()
    plt.savefig(open('Error.png','wb'))
    plt.close()

    pred_a=model.predict(a_test)    
    # Real Space NN prediction
    pred_u=pred_a@U.transpose()
    # Real Space PCA prediction (Fourier Modes)
    PCA_u=u_test@ U[:,:buildmodel.trunc()]@ U[:,:buildmodel.trunc()].transpose()

    # Plot side by side histograms
    plt.figure()
    # Plot side by side histograms
    plt.subplot(1,2,1)
    error = u_test.flatten() - pred_u.flatten()
    MSE=plot_hist(error,'NN')
    plt.subplot(1,2,2)
    error = u_test.flatten() - PCA_u.flatten()
    MSE_PCA=plot_hist(error,'PCA')
    plt.tight_layout()
    plt.savefig(open('Statistics.png','wb'))
    plt.close()
    
    # Save models
    model.save_weights('model.h5')

    # Save MSE
    pickle.dump([MSE,MSE_PCA],open('MSE.p','wb'))

    ###########################################################################
    # Put into text file
    ###########################################################################
    
    # Put variance (this is MSE off the test data) epochs, MAE, MSE in a txt file for all trials
    name='../Trials.txt'
    folder=os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    #Input header if the file doesn't exist
    from decimal import Decimal
    MSEstr='%.2E' % Decimal(str(MSE))
    MSEstr_PCA='%.2E' % Decimal(str(MSE_PCA))

    if os.path.isfile(name)==False:
        newfile=open(name,'a+')
        newfile.write("\t\t\t{}\t\t\t\t{}\n".format('MSE','MSE_PCA'))
        if len(folder)<8:
            newfile.write("{}\t\t{}\t\t{}\n".format(folder,MSEstr,MSEstr_PCA))
        else:
            newfile.write("{}\t{}\t\t{}\n".format(folder,MSEstr,MSEstr_PCA))
        newfile.close() 
    else:
        newfile=open(name,'a+')
        if len(folder)<8:
            newfile.write("{}\t\t{}\t\t{}\n".format(folder,MSEstr,MSEstr_PCA))
        else:
            newfile.write("{}\t{}\t\t{}\n".format(folder,MSEstr,MSEstr_PCA))
        newfile.close() 
