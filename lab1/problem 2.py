#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:52:53 2020

@author: teodora
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import pdb
import random

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Robbing:

    # Actions
    STAY       = 4
    MOVE_LEFT  = 3
    MOVE_RIGHT = 1
    MOVE_UP    = 2
    MOVE_DOWN  = 0

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = 0
    ROB_REWARD = 10
    IMPOSSIBLE_REWARD = -100
    CAUGHT_REWARD = -50

    def __init__(self, city, ps):
        """ Constructor of the environment Maze.
        """
        self.city                     = city;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states(); #map between row col and index
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards();
        self.police_start = ps

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;

        for i in range(self.city.shape[0]):
            for j in range(self.city.shape[1]):
                for k in range(self.city.shape[0]):
                    for l in range(self.city.shape[1]):
                        states[s] = (i,j,k,l);
                        map[(i,j,k,l)] = s;
                        s += 1;

        return states, map

    def __move(self, state, action, police_action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """

        # Compute the future position given current (state, action

        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];

        row_p = self.states[state][2] + police_action[0];
        col_p = self.states[state][3] + police_action[1];


        # Is the future position an impossible one ?
        hitting_boundaries =  (row == -1) or (row == (self.city.shape[0])) or \
                              (col == -1) or (col == (self.city.shape[1])) or \
                              (self.city[row,col] == 1);
        # Based on the impossiblity check return the next state)
        if hitting_boundaries:
            #print('hitting state', row, col, row_p, col_p)
            return self.map[(self.states[state][0], self.states[state][1], row_p, col_p)];
        else:
            #print('new state , ', row, col, row_p, col_p)
            return self.map[(row, col, row_p, col_p)];

    def __actions_police(self, state): # OK

        row = self.states[state][0];
        col = self.states[state][1];
        row_p = self.states[state][2];
        col_p = self.states[state][3];
        actions = dict();
        

        if (row_p == 0):
            #if the police is on row 0, it can always move down 
            actions[self.MOVE_DOWN]  = (1,0);

            if (col_p==0):
                #if the police is in the top left corner, it can always move right and down
                actions[self.MOVE_RIGHT] = (0, 1);
            elif (col_p == self.city.shape[1]-1):
                #if the police is in the top right column, it can always move left and down
                actions[self.MOVE_LEFT]  = (0,-1);
            else:
                #if the police is on the first row but not in the corners, its moves depend on the robber's position
                if col==col_p:
                    actions[self.MOVE_RIGHT] = (0, 1);
                    actions[self.MOVE_LEFT]  = (0,-1);
                elif col>col_p:
                    actions[self.MOVE_RIGHT] = (0, 1);
                elif col<col_p:
                    actions[self.MOVE_LEFT] = (0, -1);

        elif (row_p == self.city.shape[0]-1):
            #if the police is on the bottom row, it can always move up
            actions[self.MOVE_UP]  = (-1,0);

            if (col_p==0):
                #if the police is on the bottom left corner, it can always move right and up
                actions[self.MOVE_RIGHT] = (0, 1);
            elif (col_p == self.city.shape[1]-1):
                #if the police is on the bottom right corner, it can always move left and up
                actions[self.MOVE_LEFT]  = (0,-1);
            else:
                #if the police is on the bottom row but not in the corners, its moves depend on the robber's position
                if col==col_p:
                    actions[self.MOVE_RIGHT] = (0, 1);
                    actions[self.MOVE_LEFT]  = (0,-1);
                elif col>col_p:
                    actions[self.MOVE_RIGHT] = (0, 1);
                elif col<col_p:
                    actions[self.MOVE_LEFT] = (0, -1);

        elif ( col_p == 0):
            #if the police is on the left column, it can always move right
            actions[self.MOVE_RIGHT] = (0, 1);
            if row==row_p:
                actions[self.MOVE_UP]    = (-1,0);
                actions[self.MOVE_DOWN]  = (1,0);
            if row>row_p:
                actions[self.MOVE_UP]    = (-1,0);
            if row<row_p:
                actions[self.MOVE_DOWN] = (1,0)

        elif (col_p == self.city.shape[1]-1):
            #if the police is on the right column, it can always move left
            if row==row_p:
                actions[self.MOVE_UP]    = (-1,0);
                actions[self.MOVE_DOWN]  = (1,0);
            if row>row_p:
                actions[self.MOVE_UP]    = (-1,0);
            if row<row_p:
                actions[self.MOVE_DOWN] = (1,0)

        else:
            #if the police is in the city but not on the boundaries, its moves depend on the position of the robber
            if row==row_p:
                if col<col_p:    
                    actions[self.MOVE_LEFT]  = (0,-1);
                    actions[self.MOVE_UP]    = (-1,0);
                    actions[self.MOVE_DOWN]  = (1,0);
                if col>col_p:
                    actions[self.MOVE_RIGHT] = (0, 1);
                    actions[self.MOVE_UP]    = (-1,0);
                    actions[self.MOVE_DOWN]  = (1,0);
            if col==col_p:
                if row<row_p:
                    actions[self.MOVE_DOWN]  = (1,0);
                    actions[self.MOVE_LEFT]  = (0,-1);
                    actions[self.MOVE_RIGHT] = (0, 1);
                if row>row_p:
                    actions[self.MOVE_UP]    = (-1,0);
                    actions[self.MOVE_LEFT]  = (0,-1);
                    actions[self.MOVE_RIGHT] = (0, 1);
            else:
                if row>row_p:
                    actions[self.MOVE_UP]    = (-1,0);
                if row<row_p:
                    actions[self.MOVE_DOWN]  = (1,0);
                if col>col_p:
                    actions[self.MOVE_RIGHT] = (0, 1);
                if col<col_p:
                    actions[self.MOVE_LEFT]  = (0,-1);
        
        return actions


    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                police_moves = self.__actions_police(s)

                for p in police_moves:
                    police_action = police_moves[p]
                    next_s = self.__move(s, a, police_action);
                    if self.states[s][0]==self.states[s][2] and self.states[s][1]==self.states[s][3]:
                        if self.states[next_s][0]==0 and self.states[next_s][1]==0 and self.states[next_s][2]==self.police_start[0] and self.states[next_s][3]==self.police_start[1]:
                            transition_probabilities[next_s, s, a]=1
                    
                    else:
                        transition_probabilities[next_s, s, a] = 1/len(police_moves);
        return transition_probabilities;

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions));

        for s in range(self.n_states):
            for a in range(self.n_actions):
                police_moves = self.__actions_police(s)
                len_act_p=len(police_moves)

                for p in police_moves:
                    police_action = police_moves[p]

                    next_s = self.__move(s,a, police_action);

                    next_s_inds = self.states[next_s]

                    #Reward for being eaten by the Minotaur
                    if next_s_inds[0] == next_s_inds[2] and next_s_inds[1] == next_s_inds[3]:
                        rewards[s,a] += 1/len_act_p*self.CAUGHT_REWARD;
                    
                    # Reward for hitting the boundaries
                    elif  a != self.STAY and next_s_inds[0]== self.states[s][0] and next_s_inds[1]== self.states[s][1]:
                        rewards[s,a] += 1/len_act_p*self.IMPOSSIBLE_REWARD;
                    
                    # Reward for robbing a bank
                    elif self.city[next_s_inds[0],next_s_inds[1]] == 2:
                        rewards[s,a] += 1/len_act_p*self.ROB_REWARD;


                    # Reward for taking a step to an empty cell and not being caught
                    else:
                        rewards[s,a] += 1/len_act_p* self.STEP_REWARD;
        return rewards;