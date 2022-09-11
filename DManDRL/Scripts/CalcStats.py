#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:48:41 2019

@author: kevin
"""

import numpy as np

class Calculate:
    def __init__(self):
        self.N = 64
        self.Nh = self.N/2
        self.d = 22
        self.k = (2*np.pi/self.d) * np.concatenate((np.arange(self.Nh),np.array([0.0]),np.arange(-self.Nh+1,0)))
        self.ik = 1j * self.k
        self.ik2 = np.square(self.ik)
        self.ik = np.reshape(self.ik,[self.N,])
        self.ik2 = np.reshape(self.ik2,[self.N,])
        
    def Calc(self,u): #u
        #v = a[0::2] + 1j*a[1::2]
        #multi = (-1j*2*np.pi/N)*(-1*m)*self.k
        #multi = np.exp(multi)
        #shifted_v = np.multiply(v,multi)
        
        #x = np.real(shifted_v)
        #y = np.imag(shifted_v)
        #a_shifted = np.empty_like(a)
        #a_shifted[0::2] = x
        #a_shifted[1::2] = y
        nt = len(u[0]) #u
        u_x = np.zeros((self.N,nt))
        u_xx = np.zeros((self.N,nt))
        for i in range(nt):
            #FFT_u = aa[:,i]
            FFT_u = np.fft.fft(u[:,i])
            #print(self.ik.shape)
            #print(FFT_u.shape)
            ux = np.multiply(self.ik,FFT_u)
            #print(ux.shape)
            uxx = np.multiply(self.ik2,FFT_u)
            #print(uxx.shape)
            u_x[:,i] = np.fft.ifft(ux)
            u_xx[:,i] = np.fft.ifft(uxx)
            #u_x[:,i] = np.fft.ifft(np.reshape(ux,[self.N,]))
            #uxx[:,i] = np.fft.ifft(np.reshape(uxx,[self.N,]))
        return u_x, u_xx