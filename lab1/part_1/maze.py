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
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100
    DIE_REWARD = -99

    def __init__(self, maze, minotaur_can_stay = False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.minotaur_can_stay        = minotaur_can_stay;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states(); #map between row col and index
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards();



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

        states[s] = "Terminal";
        map["Terminal"] = s;
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

    def __move(self, state, action, minotaur_action, terminal = False):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.
            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """

        # Compute the future position given current (state, action
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];

        row_m = self.states[state][2] + minotaur_action[0];
        col_m = self.states[state][3] + minotaur_action[1];


        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == (self.maze.shape[0])) or \
                              (col == -1) or (col == (self.maze.shape[1])) or \
                              (self.maze[row,col] == 1);
        # Based on the impossiblity check return the next state)
        if hitting_maze_walls:
            #print('hitting state', row, col, row_m, col_m)
            return self.map[(self.states[state][0], self.states[state][1], row_m, col_m)];
        else:
            #print('new state , ', row, col, row_m, col_m)
            return self.map[(row, col, row_m, col_m)];

    def __actions_minotaur(self, state): # OK

        row_m = self.states[state][2];
        col_m = self.states[state][3];
        actions = dict();

        if self.minotaur_can_stay:
            actions[self.STAY]  = (0,0);


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

                if "Terminal" == self.states[s]:
                    if a == self.STAY:
                        transition_probabilities[s, s, a] = 1;
                    else:
                        transition_probabilities[:, s, a] = 0;
                else:
                    minotaur_moves = self.__actions_minotaur(s)
                    for m in minotaur_moves:
                        minotaur_action = minotaur_moves[m]
                        next_s = self.__move(s, a, minotaur_action);
                        transition_probabilities[next_s, s, a] = 1/len(minotaur_moves);



        return transition_probabilities;

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions));

        for s in range(self.n_states):
            for a in range(self.n_actions):

                if "Terminal" == self.states[s]:
                    if a == self.STAY:
                        rewards[s,a] = 0
                    else:
                        rewards[s,a] = -1
                else:

                    minotaur_moves = self.__actions_minotaur(s)
                    len_act_m=len(minotaur_moves)

                    for m in minotaur_moves:
                        minotaur_action = minotaur_moves[m]

                        next_s = self.__move(s,a, minotaur_action);

                        next_s_inds = self.states[next_s]

                        #Reward for being eaten by the Minotaur
                        if next_s_inds[0] == next_s_inds[2] and next_s_inds[1] == next_s_inds[3]:
                            rewards[s,a] += 1/len_act_m*self.DIE_REWARD;

                        # Reward for hitting a wall
                        elif  a != self.STAY and next_s_inds[0]== self.states[s][0] and next_s_inds[1]== self.states[s][1]:
                            rewards[s,a] += 1/len_act_m*self.IMPOSSIBLE_REWARD;

                        # Reward for reaching the exit
                        elif self.maze[next_s_inds[0],next_s_inds[1]] == 2:
                            rewards[s,a] += 1/len_act_m*self.GOAL_REWARD;

                        # Reward for taking a step to an empty cell that is not the exit
                        else:
                            rewards[s,a] += 1/len_act_m * self.STEP_REWARD;
        return rewards;

    def goal_reached(self, s):
        state_indicies = self.states[s];
        return self.maze[state_indicies[0],state_indicies[1]] == 2 and self.maze[state_indicies[2],state_indicies[3]] != 2


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

            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1 and not self.goal_reached(s):

                minotaur_moves = self.__actions_minotaur(s)
                m =  random.sample(list(minotaur_moves), 1)[0]
                minotaur_action = minotaur_moves[m]
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s,t], minotaur_action);
                # Add the position in the maze corresponding to the next state
                # to the path

                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;

        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            minotaur_moves = self.__actions_minotaur(s)
            m =  random.sample(list(minotaur_moves), 1)[0]
            minotaur_action = minotaur_moves[m]

            # Move to next state given the policy and the current state
            next_s = self.__move(s, policy[s], minotaur_action);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while not self.goal_reached(s):# s != next_s: #
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                minotaur_moves = self.__actions_minotaur(s)
                m =  random.sample(list(minotaur_moves), 1)[0]
                minotaur_action = minotaur_moves[m]

                # Move to next state given the policy and the current state

                next_s = self.__move(s, policy[s], minotaur_action);
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

    #grid.auto_set_font_size(False)
    #grid.set_fontsize(12)

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

        if i >= 0:
            player_pos_text = grid.get_celld()[player_pos].get_text().get_text()
            minotaur_pos_text = grid.get_celld()[minotaur_pos].get_text().get_text()
            #print(minotaur_pos_text)
            #pdb.set_trace()


            if player_pos_text == '':
                #grid.get_celld()[player_pos].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[player_pos].get_text().set_text('P ' + str(i))
            else:
                if 'P' in player_pos_text and 'M' in player_pos_text:
                    text_split = player_pos_text.split('\n')
                    grid.get_celld()[player_pos].get_text().set_text(text_split[0] +', ' + str(i) +'\n'+ text_split[1])
                    grid.get_celld()[player_pos].get_text().set_color('purple')

                elif 'P' in player_pos_text:
                    grid.get_celld()[player_pos].get_text().set_text(player_pos_text +', ' + str(i))
                else:
                    #text_split = player_pos_text.split('\n')
                    grid.get_celld()[player_pos].get_text().set_text('P ' + str(i) +'\n'+ player_pos_text)
                    grid.get_celld()[player_pos].get_text().set_color('purple')

            if minotaur_pos_text == '':
                #grid.get_celld()[player_pos].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[minotaur_pos].get_text().set_text('M ' + str(i))
                grid.get_celld()[minotaur_pos].get_text().set_color('red')
            else:
                if 'P' in minotaur_pos_text and 'M' in minotaur_pos_text:
                    text_split = minotaur_pos_text.split('\n')
                    grid.get_celld()[minotaur_pos].get_text().set_text(text_split[0] +'\n'+ text_split[1]+ ', ' +str(i))
                    grid.get_celld()[minotaur_pos].get_text().set_color('purple')

                elif 'M' in minotaur_pos_text:
                    grid.get_celld()[minotaur_pos].get_text().set_text(minotaur_pos_text +', ' + str(i))
                    grid.get_celld()[minotaur_pos].get_text().set_color('red')

                else: # aleady a P there
                    #text_split = player_pos_text.split('\n')
                    grid.get_celld()[player_pos].get_text().set_text( minotaur_pos_text +'\n' +' M ' + str(i) )
                    grid.get_celld()[minotaur_pos].get_text().set_color('purple')


        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
    plt.savefig('Q1_a_optimal_path.png')




if __name__ == '__main__':
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

    gamma = 29/30
    eps = 0.5
    # Solve the MDP problem with value iteration
    #print('start val it')
    #V, policy= value_iteration(env,gamma,eps);
    #print('done val it')
    # Simulate the shortest path starting from position A
    #method = 'ValIter';
    #start  = (0,0,6,5);
    #path = env.simulate(start, policy, method);
    #print(path)
    #input('stopp')

    # Dynamic programming
    horizon = 20
    # Solve the MDP problem with dynamic programming
    V, policy= dynamic_programming(env, horizon)
    #pdb.set_trace()
    # Simulate the shortest path starting from position A
    method = 'DynProg';
    start  = (0,0,6,5);
    path = env.simulate(start, policy, method)
    print(path)
    # Show the shortest path
    animate_solution(maze, path)
