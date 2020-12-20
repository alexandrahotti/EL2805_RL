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
from problem_1 import MyNetwork
from mpl_toolkits.mplot3d import Axes3D


def opt_pol_analysis(Q_theta, max_type):
    
    #y and w values used for the plot 
    y=[0.15*k for k in range(11)]
    
    step_omega=2*pi/10
    omega=[-pi+step_omega*k for k in range(11)]
    
    #create a state list 
    states=[]
    
    #create a matrix state_matrix where state_matrix[y,w] corresponds to the index of the state corresponding to y,w in the state list 
    states_matrix=np.array([[0 for i in range(11)] for j in range(11)])
    
    #lists used for the plot
    y_qmax_list=[]
    omega_qmax_list=[]
    y_amax_list=[]
    omega_amax_list=[]
    k=0
    
    #fill the state list and fill the state_matrix with the indices 
    for i in range(11):
        for j in range(11):
            states.append((0,y[i],0,0,omega[j],0,0,0))
            states_matrix[i][j]=k
            k+=1
            
    states_tensor=torch.tensor(states, requires_grad=False, dtype=torch.float32)
    
    
    
    #If we want to plot the Q values 
    if max_type=="Q":
        Q_max=[]
        for i in range(121):
            #compute the max value for every state
            max_value=float(torch.max(Q_theta(states_tensor)[i]))
            Q_max.append(max_value)
            
            #search the y,w corresponding to the state for which we computed the max value 
            y_qmax, omega_qmax=np.where(states_matrix==i)
            y_qmax_list.append(y[int(y_qmax)])
            omega_qmax_list.append(omega[int(omega_qmax)])
            
            
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(omega_qmax_list,y_qmax_list, Q_max)
        ax.set_xlabel('Lander angle')
        ax.set_ylabel('lander height')
        ax.set_zlabel('Max value')
        ax.set_title('Max value Q_max=f(omega, y)')
        
        print("y_qmax", y_qmax_list)
        print("omega_qmax", omega_qmax_list)
        print("q_max", Q_max)
        
        plt.show()
    
    #if we want to plot the actions
    else:
        a_max=[]
        for i in range(121):
            max_action=int(torch.argmax(Q_theta(states_tensor)[i]))
            #print(max_action)
            a_max.append(max_action)
            y_amax, omega_amax=np.where(states_matrix==i)
            #print(y_amax, omega_amax)
            y_amax_list.append(y[int(y_amax)])
            omega_amax_list.append(omega[int(omega_amax)])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(omega_amax_list, y_amax_list, a_max)
        ax.set_xlabel('Lander angle')
        ax.set_ylabel('lander height')
        ax.set_zlabel('optimal action')
        ax.set_title('Optimal action a_max=f(omega, y)')
        plt.show() 

    
Q_theta=torch.load("neural-network-1.pth")

opt_pol_analysis(Q_theta, "Q")
    



    