##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Fri Dec 18 15:19:18 2020
#
#@author: teodora
#"""
#
import numpy as np
import gym
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from DQN_agent import RandomAgent
import random
import pdb
import tqdm
import matplotlib.pyplot as plt
from math import pi
from problem_1_working import MyNetwork
from mpl_toolkits.mplot3d import Axes3D


def opt_pol_analysis(Q_theta, max_type):
    y=[0.15*k for k in range(10)]
    
    step_omega=2*pi/10
    omega=[-pi+step_omega*k for k in range(10)]
    
    states=[]
    states_matrix=np.array([[0 for i in range(10)] for j in range(10)])
    
    y_qmax_list=[]
    omega_qmax_list=[]
    y_amax_list=[]
    omega_amax_list=[]
    k=1
    
    for i in range(10):
        for j in range(10):
            states.append((0,y[i],0,0,omega[j],0,0,0))
            states_matrix[i][j]=k
            k+=1
            #print(states)
    states_tensor=torch.tensor(states, requires_grad=False, dtype=torch.float32)
    print(states_matrix)
    
    #print(states_tensor.shape)
    
    if max_type=="Q":
        Q_max=[]
        for i in range(1,101):
            max_value=float(torch.max(Q_theta(states_tensor)[i-1]))
            Q_max.append(max_value)
            y_qmax, omega_qmax=np.where(states_matrix==i)
            y_qmax_list.append(int(y_qmax))
            omega_qmax_list.append(int(omega_qmax))
            
            
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(y_qmax_list, omega_qmax_list, Q_max)
        plt.show()
    
    else:
        a_max=[]
        for i in range(1,101):
            max_action=int(torch.argmax(Q_theta(states_tensor)[i-1]))
            #print(max_action)
            a_max.append(max_action)
            y_amax, omega_amax=np.where(states_matrix==i)
            #print(y_amax, omega_amax)
            y_amax_list.append(y[int(y_amax)])
            omega_amax_list.append(omega[int(omega_amax)])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(y_amax_list, omega_amax_list, a_max)
        plt.show()   
            
    
Q_theta=torch.load("first_neural-network-1.pth")

y=[0.15*k for k in range(10)]
    
step_omega=2*pi/100
omega=[-pi+step_omega*k for k in range(100)]

states=[]

for i in range(10):
        for j in range(10):
            states.append((0,y[i],0,0,omega[j],0,0,0))
            #print(states)
states_tensor=torch.tensor(states, requires_grad=False, dtype=torch.float32)
#print(Q_theta(states_tensor)[88])
opt_pol_analysis(Q_theta, "Q")
    
#x=np.array([[1,2,3,4],[5,6,7,8]])
#x,y=np.where(x==5)
#print(int(x), int(y))
#y=np.array([0,1,1,1])
#z=np.array([10,8,7,10])
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot(x,y,z)
#plt.show()


    