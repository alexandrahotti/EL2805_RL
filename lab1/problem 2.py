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

    def __init__(self, city):
        """ Constructor of the environment city.
        """
        self.city                     = city;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states(); #map between row col and index
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards();
        #self.police_start             = ps;

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
        """ Makes a step in the city, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the city that agent transitions to.
        """
        #If the robber is already caught, return initial position
        if self.states[state][0]== self.states[state][2] and self.states[state][1]==self.states[state][3]:
            return self.map[(0,0,1,2)]
        
        # Compute the future position given current (state, action)
        else:
            row = self.states[state][0] + self.actions[action][0];
            col = self.states[state][1] + self.actions[action][1];

            row_police = self.states[state][2] + police_action[0];
            col_police = self.states[state][3] + police_action[1];


        
            # Is the future position an impossible one (given the boundaries)?
            hitting_boundaries =  (row == -1) or (row == (self.city.shape[0])) or \
                                  (col == -1) or (col == (self.city.shape[1])) 
        
        
            # Based on the impossiblity check return the next state
            if hitting_boundaries:
                return self.map[(self.states[state][0], self.states[state][1], row_police, col_police)];
            else:
                return self.map[(row, col, row_police, col_police)];

    def __actions_police(self, state): # OK

        row = self.states[state][0];
        col = self.states[state][1];
        row_police = self.states[state][2];
        col_police = self.states[state][3];
        actions = dict();
        

        if (row_police == 0):
            #if the police is on row 0 (top row), it can always move down 
            actions[self.MOVE_DOWN]  = (1,0);

            if (col_police==0):
                #if the police is in the top left corner, it can always move right and down
                actions[self.MOVE_RIGHT] = (0, 1);
            elif (col_police == self.city.shape[1]-1):
                #if the police is in the top right column, it can always move left and down
                actions[self.MOVE_LEFT]  = (0,-1);
            else:
                #if the police is on the top row but not in the corners, its moves depend on the robber's position
                if col==col_police:
                    actions[self.MOVE_RIGHT] = (0, 1);
                    actions[self.MOVE_LEFT]  = (0,-1);
                elif col>col_police:
                    actions[self.MOVE_RIGHT] = (0, 1);
                elif col<col_police:
                    actions[self.MOVE_LEFT] = (0, -1);

        elif (row_police == self.city.shape[0]-1):
            #if the police is on the bottom row, it can always move up
            actions[self.MOVE_UP]  = (-1,0);

            if (col_police==0):
                #if the police is on the bottom left corner, it can always move right and up
                actions[self.MOVE_RIGHT] = (0, 1);
            elif (col_police == self.city.shape[1]-1):
                #if the police is on the bottom right corner, it can always move left and up
                actions[self.MOVE_LEFT]  = (0,-1);
            else:
                #if the police is on the bottom row but not in the corners, its moves depend on the robber's position
                if col==col_police:
                    actions[self.MOVE_RIGHT] = (0, 1);
                    actions[self.MOVE_LEFT]  = (0,-1);
                elif col>col_police:
                    actions[self.MOVE_RIGHT] = (0, 1);
                elif col<col_police:
                    actions[self.MOVE_LEFT] = (0, -1);

        elif (col_police == 0):
            #if the police is on the left column (the corners are excluded here because they have already been handled), it can always move right
            actions[self.MOVE_RIGHT] = (0, 1);
            if row==row_police:
                actions[self.MOVE_UP]    = (-1,0);
                actions[self.MOVE_DOWN]  = (1,0);
            if row<row_police: #the robber is above the police
                actions[self.MOVE_UP]    = (-1,0);
            if row>row_police: #the robber is below the police
                actions[self.MOVE_DOWN] = (1,0)

        elif (col_police == self.city.shape[1]-1):
            #if the police is on the right column, it can always move left
            if row==row_police:
                actions[self.MOVE_UP]    = (-1,0);
                actions[self.MOVE_DOWN]  = (1,0);
            if row<row_police:
                actions[self.MOVE_UP]    = (-1,0);
            if row>row_police:
                actions[self.MOVE_DOWN] = (1,0)

        else:
            #if the police is in the city but not on the boundaries, its moves depend on the position of the robber
            if row==row_police:
                if col<col_police:    
                    actions[self.MOVE_LEFT]  = (0,-1);
                    actions[self.MOVE_UP]    = (-1,0);
                    actions[self.MOVE_DOWN]  = (1,0);
                if col>col_police:
                    actions[self.MOVE_RIGHT] = (0, 1);
                    actions[self.MOVE_UP]    = (-1,0);
                    actions[self.MOVE_DOWN]  = (1,0);
            elif col==col_police:
                if row>row_police:
                    actions[self.MOVE_DOWN]  = (1,0);
                    actions[self.MOVE_LEFT]  = (0,-1);
                    actions[self.MOVE_RIGHT] = (0, 1);
                if row<row_police:
                    actions[self.MOVE_UP]    = (-1,0);
                    actions[self.MOVE_LEFT]  = (0,-1);
                    actions[self.MOVE_RIGHT] = (0, 1);
            else:
                if row<row_police:
                    actions[self.MOVE_UP]    = (-1,0);
                if row>row_police:
                    actions[self.MOVE_DOWN]  = (1,0);
                if col>col_police:
                    actions[self.MOVE_RIGHT] = (0, 1);
                if col<col_police:
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
                    
                    if self.states[s][0]==self.states[s][2] and self.states[s][1]==self.states[s][3]: #if the robber is caught in state s, next_s is always the initial position
                        transition_probabilities[next_s, s, a]=1
                    
                    else: #if the robber is not caught is state s, next_s is reached with a probability of 1/len(police_moves)
                        transition_probabilities[next_s, s, a] = 1/len(police_moves);
        return transition_probabilities;

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions));

        for s in range(self.n_states):
            for a in range(self.n_actions):
                
                #If the robber is already caught, the only possibility is to get back to the initial position. The reward is 0 because he does not looses anything (he already lost his money when he got caught)
                if self.states[s][0]==self.states[s][2] and self.states[s][1]==self.states[s][3]: 
                    rewards[s,a]=self.STEP_REWARD
                
                #If the robber was not caught, he may get caught or rob a bank at the next state and his reward depends on the action of the police
                else:
                    police_moves = self.__actions_police(s)
                    len_act_p=len(police_moves)
    
                    for p in police_moves:
                        police_action = police_moves[p]
                        next_s = self.__move(s,a, police_action);
                        next_s_inds = self.states[next_s]
    
                        #Reward for being caught by the police
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

    def simulate(self, start, V, policy):

        path = list();
    #        if method == 'DynProg':
    #            # Deduce the horizon from the policy shape
    #            horizon = policy.shape[1];
    #            # Initialize current state and time
    #            t = 0;
    #            s = self.map[start];
    #
    #            # Add the starting position in the maze to the path
    #            path.append(start);
    #            while t < horizon-1 and not self.goal_reached(s):
    #
    #                minotaur_moves = self.__actions_minotaur(s)
    #
    #                m =  random.sample(list(minotaur_moves), 1)[0]
    #
    #                minotaur_action = minotaur_moves[m]
    #
    #
    #
    #                # Move to next state given the policy and the current state
    #                next_s = self.__move(s, policy[s,t], minotaur_action);
    #                # Add the position in the maze corresponding to the next state
    #                # to the path
    #
    #
    #
    #                path.append(self.states[next_s])
    #                # Update time and state for next iteration
    #                t +=1;
    #                s = next_s;
    
            # Initialize current state, next state and time
        t = 1;
        s = self.map[start];
        r=0 #reward collected so far
        # Add the starting position in the city to the path
        path.append(start);
        
        #Select the action of the police
        police_moves = self.__actions_police(s)
        p =  random.sample(list(police_moves), 1)[0]
        police_action = police_moves[p]
        
        # Move to next state given the policy and the current state
        next_s = self.__move(s,policy[s], police_action);
        
        #Update the reward
        #If getting caught
        if self.states[next_s][0]==self.states[next_s][2] and self.states[next_s][1]==self.states[next_s][3]:
            r+=self.CAUGHT_REWARD
            
        # Reward for hitting the boundaries
        elif  policy[s] != self.STAY and self.states[next_s][0]== self.states[s][0] and self.states[next_s][1]== self.states[s][1]:
            r += self.IMPOSSIBLE_REWARD;
                        
        # Reward for robbing a bank
        elif self.city[self.states[next_s][0],self.states[next_s][1]] == 2:
            r += self.ROB_REWARD;

        # Reward for taking a step to an empty cell and not being caught
        else:
            r += self.STEP_REWARD;
        
        print(r)
        # Add the position in the maze corresponding to the next state
        # to the path
        path.append(self.states[next_s]);
        # Loop while state is not the goal state
        while r<V[0]:
            # Update state
            s = next_s;
            
            #Select the action of the police
            police_moves = self.__actions_police(s)
            p =  random.sample(list(police_moves), 1)[0]
            police_action = police_moves[p]
        
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s], police_action);
            #Update the reward
            #If getting caught
            if self.states[next_s][0]==self.states[next_s][2] and self.states[next_s][1]==self.states[next_s][3]:
                r+=self.CAUGHT_REWARD
            
            # Reward for hitting the boundaries
            elif  policy[s] != self.STAY and self.states[next_s][0]== self.states[s][0] and self.states[next_s][1]== self.states[s][1]:
                r += self.IMPOSSIBLE_REWARD;
                            
            # Reward for robbing a bank
            elif self.city[self.states[next_s][0],self.states[next_s][1]] == 2:
                r += self.ROB_REWARD;
    
            # Reward for taking a step to an empty cell and not being caught
            else:
                r += self.STEP_REWARD;
                
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Update time and state for next iteration
            t +=1;
            print(r)
        return path

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input city env           : The city environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space

    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

def draw_city(city):

    # Map a color to each cell in the city
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN};

    # Give a color to each cell
    rows,cols    = city.shape;
    colored_city = [[col_map[city[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The City');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_city,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    
def animate_solution(city, path):

    # Map a color to each cell in the city
    col_map = {0: WHITE, 2: LIGHT_GREEN};

    # Size of the city
    rows,cols = city.shape;

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_city = [[col_map[city[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the city
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_city,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);



    # Update the color at each frame
    for i in range(len(path)):

        player_pos = (path[i][:2])
        police_pos = (path[i][2:])

        player_past_pos = (path[i-1][:2])
        police_past_pos = (path[i-1][2:])

        grid.get_celld()[player_pos].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[player_pos].get_text().set_text('Robber')
        
        grid.get_celld()[police_pos].set_facecolor(LIGHT_RED)
        grid.get_celld()[police_pos].get_text().set_text('Police')
        
        if i > 0:
            if player_pos == police_pos:
                 grid.get_celld()[player_pos].set_facecolor(LIGHT_RED)
                 grid.get_celld()[player_pos].get_text().set_text('PLAYER CAUGHT')
        
            if player_pos == player_past_pos:
                grid.get_celld()[player_pos].set_facecolor(LIGHT_GREEN)
                #grid.get_celld()[player_pos].get_text().set_text('Player is out')
            else:
                grid.get_celld()[player_past_pos].set_facecolor(col_map[city[player_past_pos]])
                grid.get_celld()[player_past_pos].get_text().set_text('')

            if police_pos == police_past_pos:
                grid.get_celld()[police_pos].set_facecolor(LIGHT_RED)
                #grid.get_celld()[police_pos].get_text().set_text('M is out')
            else:
                grid.get_celld()[police_past_pos].set_facecolor(col_map[city[police_past_pos]])
                grid.get_celld()[police_past_pos].get_text().set_text('')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)




def main():
    city = np.array([
        [2,0,0,0,0,2],
        [0,0,0,0,0,0],
        [2,0,0,0,0,2]
    ])


    # Create an environment city
    env = Robbing(city)
    #draw_city(city)

    epsilon=0.0001
    list_values=[]
    for i in range (5, 100, 5):
        l=i*0.01
        V, policy=value_iteration(env, l, epsilon)
        list_values.append(V[0])
    list_lambda=[i*0.01 for i in range(5, 100, 5)]
    plt.scatter(list_lambda, list_values)
    plt.xlabel("lambda")
    plt.ylabel("value")
        
    
    
    # Discount Factor 
#    gamma = 0.95; 
##    Accuracy treshold 
#    epsilon = 0.0001;
#    
#    V, policy = value_iteration(env, gamma, epsilon)
#    #print(policy)
#    #print(len(policy))
#    #print(n)
#    print(V[0])
    
    #start  = (0,0,1,2);
    #path = env.simulate(start, V, policy)
    
    #animate_solution(city, path)
    

main()