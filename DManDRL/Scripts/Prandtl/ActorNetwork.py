#! /home/linot/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:26:32 2019

@author: kevin
"""

import numpy as np
import math
from tensorflow.keras.initializers import normal, identity
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.engine.training import collect_trainable_weights
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda #merge, 
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K

HIDDEN1_UNITS = 128
HIDDEN2_UNITS = 64

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Build Actor Model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("Building Actor Network...")
        S = Input(shape=[state_size])   
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        V = Dense(action_dim,activation='tanh')(h1)
        #V = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)  
        #Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
        #Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1) 
        #V = merge([Steering,Acceleration,Brake],mode='concat')          
        model = Model(inputs=S,outputs=V)
        return model, model.trainable_weights, S