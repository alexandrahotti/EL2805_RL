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

class Maze:

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
    DIE_REWARD = -1
    TERM_REWARD = 0
    GOAL_REWARD = 1
    IMPOSSIBLE_REWARD = -100


    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states(); #map between row col and index
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);

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

        states[s] = "W";
        map["W"] = s;
        s += 1;

        states[s] = "D";
        map["D"] = s;
        s += 1;

        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):

                if self.maze[i,j] != 1:
                    for k in range(self.maze.shape[0]):
                        for l in range(self.maze.shape[1]):

                            states[s] = (i,j,k,l);
                            map[(i,j,k,l)] = s;
                            s += 1;

        return states, map

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """

        # Compute the future position given current (state, action

        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];

        row_m = self.states[state][2]; #+ minotaur_action[0];
        col_m = self.states[state][3]; #+ minotaur_action[1];


        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == (self.maze.shape[0])) or \
                              (col == -1) or (col == (self.maze.shape[1])) or \
                              (self.maze[row,col] == 1);
        # Based on the impossiblity check return the next state.
        #input()
        if hitting_maze_walls:
            #print('hitting state', row, col, row_m, col_m)
            return (self.states[state][0], self.states[state][1], row_m, col_m);
        else:
            #print('new state , ', row, col, row_m, col_m)
            return (row, col, row_m, col_m);

    def __actions_minotaur(self, state): # OK

        #print(self.states[state])
        row_m = self.states[state][2];
        col_m = self.states[state][3];
        actions = dict();

        #actions[self.STAY]  = (0,0);


        # Is the future position an impossible one ?
        if (row_m == 0):
            #print('row_m == 0')
            actions[self.MOVE_DOWN]  = (1,0);

            if (col_m==0):
                #print('col_m == 0')
                actions[self.MOVE_RIGHT] = (0, 1);
            elif (col_m == self.maze.shape[1]-1):
                #print('col_m == self.maze.shape[1]')
                actions[self.MOVE_LEFT]  = (0,-1);
            else:
                #print('else in row 0')
                actions[self.MOVE_RIGHT] = (0, 1);
                actions[self.MOVE_LEFT]  = (0,-1);

        elif (row_m == self.maze.shape[0]-1):
            #print('row_m == self.maze.shape[0]')
            actions[self.MOVE_UP]  = (-1,0);

            if (col_m==0):
                actions[self.MOVE_RIGHT] = (0, 1);
            elif (col_m == self.maze.shape[1]-1):
                actions[self.MOVE_LEFT]  = (0,-1);
            else:
                actions[self.MOVE_RIGHT] = (0, 1);
                actions[self.MOVE_LEFT]  = (0,-1);

        elif ( col_m == 0):
            #print('col_m == 0')
            actions[self.MOVE_RIGHT] = (0, 1);
            actions[self.MOVE_UP]    = (-1,0);
            actions[self.MOVE_DOWN]  = (1,0);

        elif ( col_m == self.maze.shape[1]-1):
            #print('col_m == self.maze.shape[1]')
            actions[self.MOVE_LEFT] = (0, -1);
            actions[self.MOVE_UP]    = (-1,0);
            actions[self.MOVE_DOWN]  = (1,0);

        else:
            #print('else')
            actions[self.MOVE_LEFT]  = (0,-1);
            actions[self.MOVE_RIGHT] = (0, 1);
            actions[self.MOVE_UP]    = (-1,0);
            actions[self.MOVE_DOWN]  = (1,0);

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

                if self.__is_in_terminal(self.states[s]):
                    print('got to terminal',self.states[s])
                    next_s = s
                    transition_probabilities[next_s, s, a] = 1;

                elif self.__next_is_death(self.states[s]):

                    next_s = self.map['D']
                    transition_probabilities[next_s, s, a] = 1;

                elif self.__next_is_win(self.states[s]):

                    next_s = self.map['W']
                    transition_probabilities[next_s, s, a] = 1;

                else:


                    next_s = self.__move(s, a);
                    minotaur_moves = self.__actions_minotaur(s)

                    for m in minotaur_moves:
                        m_action = minotaur_moves[m]
                        next_s_m = self.map[(next_s[0], next_s[1], next_s[2]+m_action[0], next_s[3]+m_action[1])]
                        transition_probabilities[next_s_m, s, a] = 1/len(minotaur_moves);

        return transition_probabilities;

    def __is_in_terminal(self, state):
        return state == 'D' or state == 'W'

    def __next_is_win(self, state):
        return not self.__next_is_death(state) and self.maze[state[0],state[1]] == 2

    def __next_is_death(self, state):
        return (state[0],state[1]) == (state[2],state[3])

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if self.__is_in_terminal(self.states[s]):
                    next_s = s
                    rewards[s,a] = self.TERM_REWARD
                else:

                    next_s = self.__move(s,a);

                    # Rewrd for hitting a wall
                    if s == next_s and a != self.STAY: # ok
                        rewards[s,a] = self.IMPOSSIBLE_REWARD;
                    # Reward for reaching the exit

                    elif self.__is_in_terminal(next_s):
                        rewards[s,a] = self.TERM_REWARD;

                    elif next_s[0] == next_s[2] and next_s[1] == next_s[3]:
                        rewards[s,a] = self.DIE_REWARD;

                    elif self.__next_is_win(next_s):
                        rewards[s,a] = self.GOAL_REWARD;

                    else:
                        rewards[s,a] = self.STEP_REWARD;
        return rewards;

    def goal_reached(self, s):

        return self.maze[s[0],s[1]] == 2 and self.maze[s[2],s[3]] != 2


    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            #print('s2 ', s)
            # Add the starting position in the maze to the path
            path.append(start);
            #pdb.set_trace()
            while t < horizon-1:

                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s,t]);
                # Add the position in the maze corresponding to the next state
                # to the path

                minotaur_moves = self.__actions_minotaur(s)
                a_m =  minotaur_moves[random.sample(list(minotaur_moves), 1)[0]]


                next_s = (next_s[0], next_s[1], next_s[2]+a_m[0], next_s[3]+a_m[1])

                path.append(next_s)
                # Update time and state for next iteration
                t +=1;
                s = self.map[next_s];
                #print('s3 ', s)

        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
        return path


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
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
    # - The finite horizon
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

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
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
        minotaur_pos = (path[i][2:])

        player_past_pos = (path[i-1][:2])
        minotaur_past_pos = (path[i-1][2:])

        grid.get_celld()[player_pos].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[player_pos].get_text().set_text('Player')

        grid.get_celld()[minotaur_pos].set_facecolor(LIGHT_RED)
        grid.get_celld()[minotaur_pos].get_text().set_text('M')

        if i > 0:
            if player_pos == player_past_pos:
                grid.get_celld()[player_pos].set_facecolor(LIGHT_GREEN)
                #grid.get_celld()[player_pos].get_text().set_text('Player is out')
            else:
                grid.get_celld()[player_past_pos].set_facecolor(col_map[maze[player_past_pos]])
                grid.get_celld()[player_past_pos].get_text().set_text('')

            if minotaur_pos == minotaur_past_pos:
                grid.get_celld()[minotaur_pos].set_facecolor(LIGHT_RED)
                #grid.get_celld()[minotaur_pos].get_text().set_text('M is out')
            else:
                grid.get_celld()[minotaur_past_pos].set_facecolor(col_map[maze[minotaur_past_pos]])
                grid.get_celld()[minotaur_past_pos].get_text().set_text('')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)




def main():
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]
    ])


    # Create an environment maze
    env = Maze(maze)

    #env.show()
    # Finite horizon
    horizon = 20
    # Solve the MDP problem with dynamic programming
    V, policy= dynamic_programming(env, horizon)
    #pdb.set_trace()
    # Simulate the shortest path starting from position A
    method = 'DynProg';
    start  = (0,0,6,5);
    path = env.simulate(start, policy, method)
    #print(path)
    #input()

    # Show the shortest path
    animate_solution(maze, path)


main()
