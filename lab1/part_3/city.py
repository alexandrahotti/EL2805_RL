import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import pdb
import random


# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class City:

    # Actions
    MOVE_LEFT  = 0
    MOVE_RIGHT = 1
    MOVE_UP    = 2
    MOVE_DOWN  = 3
    STAY       = 4

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
    ROBBING_REWARD = 1
    CAUGHT_REWARD = -10
    IMPOSSIBLE_REWARD = -100

    def __init__(self, city):
        """ Constructor of the environment Maze.
        """
        self.city                     = city;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states(); #map between row col and index
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
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

        for i in range(self.city.shape[0]):
            for j in range(self.city.shape[1]):
                for k in range(self.city.shape[0]):
                    for l in range(self.city.shape[1]):
                            states[s] = (i,j,k,l);
                            map[(i,j,k,l)] = s;
                            s += 1;
        return states, map

    def move(self, state, action, minotaur_action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.
            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """

        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];

        row_m = self.states[state][2] + minotaur_action[0];
        col_m = self.states[state][3] + minotaur_action[1];


        hitting_walls =  (row == -1) or (row == (self.city.shape[0])) or \
                              (col == -1) or (col == (self.city.shape[1]));
        # Based on the impossiblity check return the next state)
        if hitting_walls:
            return self.map[(self.states[state][0], self.states[state][1], row_m, col_m)];
        else:
            return self.map[(row, col, row_m, col_m)];

    def actions_police(self, state):

        row_m = self.states[state][2];
        col_m = self.states[state][3];
        actions = dict();


        if (row_m == 0):

            actions[self.MOVE_DOWN]  = (1,0);

            if (col_m==0):
                actions[self.MOVE_RIGHT] = (0, 1);
            elif (col_m == self.city.shape[1]-1):
                actions[self.MOVE_LEFT]  = (0,-1);
            else:
                actions[self.MOVE_RIGHT] = (0, 1);
                actions[self.MOVE_LEFT]  = (0,-1);

        elif (row_m == self.city.shape[0]-1):
            actions[self.MOVE_UP]  = (-1,0);

            if (col_m==0):
                actions[self.MOVE_RIGHT] = (0, 1);
            elif (col_m == self.city.shape[1]-1):
                actions[self.MOVE_LEFT]  = (0,-1);
            else:
                actions[self.MOVE_RIGHT] = (0, 1);
                actions[self.MOVE_LEFT]  = (0,-1);

        elif ( col_m == 0):
            #print('col_m == 0')
            actions[self.MOVE_RIGHT] = (0, 1);
            actions[self.MOVE_UP]    = (-1,0);
            actions[self.MOVE_DOWN]  = (1,0);

        elif ( col_m == self.city.shape[1]-1):
            actions[self.MOVE_LEFT] = (0, -1);
            actions[self.MOVE_UP]    = (-1,0);
            actions[self.MOVE_DOWN]  = (1,0);

        else:
            actions[self.MOVE_LEFT]  = (0,-1);
            actions[self.MOVE_RIGHT] = (0, 1);
            actions[self.MOVE_UP]    = (-1,0);
            actions[self.MOVE_DOWN]  = (1,0);

        return actions


    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions));

        for s in range(self.n_states):
            for a in range(self.n_actions):
                police_moves = self.actions_police(s)

                for p in police_moves:
                    police_action = police_moves[p]
                    next_s = self.states[self.move(s,a,police_action)]

                    #Reward for being eaten by the Minotaur
                    if next_s[0] == next_s[2] and next_s[1] == next_s[3]:
                        rewards[s,a] = self.CAUGHT_REWARD;

                    # Reward for hitting a wall
                    elif  a != self.STAY and next_s[0] == self.states[s][0] and next_s[1] == self.states[s][1]:
                        rewards[s,a] = self.IMPOSSIBLE_REWARD;

                    elif self.city[next_s[0],next_s[1]] == 2:
                        rewards[s,a] = self.ROBBING_REWARD;

                    else:
                        rewards[s,a] = self.STEP_REWARD;
        return rewards;



def sample_eps_greedy_action(Q, s, eps):

    rn = np.random.uniform(0,1)

    if rn < eps:
        # should sample uniformly here
        return random.sample(range(0,len(Q[s,:])), 1)[0]
    else:
        return np.argmax(Q[s,:])


def sample_random_action(Q,s):

    return random.sample(range(0,len(Q[s,:])), 1)[0]


def SARSA(env, eps, gamma, start_state, no_iterations):

    r   = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    states  = env.states;
    actions = env.actions;

    Q_initial_state = np.zeros(no_iterations)

    Q  = np.zeros((n_states, n_actions));
    nv  = np.zeros((n_states, n_actions)); # no visits of (s,a)
    pi  = np.random.randint(n_actions, size = n_states ); # Randomly initilize the policy

    s = env.map[start_state]
    a = sample_eps_greedy_action(Q, s, eps)

    for t in range(1, no_iterations):

        nv[s,a] += 1 # not really using this right now. maybe same as before

        alpha = 1/(nv[s,a]**(2/3))

        # Generate next r_t, s_t+1 and a_t+1
        police_moves = env.actions_police(s)
        p =  random.sample(list(police_moves), 1)[0]
        police_action = police_moves[p]

        next_s = env.move(s, a, police_action)
        next_a = sample_eps_greedy_action(Q, next_s, eps)

        Q[s,a] = Q[s,a] + alpha * (r[s,a] + gamma * Q[next_s, next_a] - Q[s,a])
        Q_initial_state[t] = np.max(Q[env.map[start_state]])

        s = next_s
        a = next_a


    return Q, Q_initial_state


def Q_learning(env, gamma, epsilon, no_iterations, start_state):

    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    states  = env.states;
    actions = env.actions;

    Q  = np.zeros((n_states, n_actions));
    nv  = np.zeros((n_states, n_actions)); # no visits of (s,a)

    s = env.map[start_state]
    a = sample_random_action(Q,s)

    Q_initial_state = np.zeros(no_iterations)


    for t in range(1, no_iterations):
        #lr = 1/t # remove?

        nv[s,a] += 1

        police_moves = env.actions_police(s)
        p =  random.sample(list(police_moves), 1)[0]
        police_action = police_moves[p]

        next_s = env.move(s, a, police_action)

        alpha = 1/(nv[s,a]**(2/3))

        Q[s,a] = Q[s,a] + alpha * (r[s,a] + gamma * max(Q[next_s]) - Q[s,a])
        Q_initial_state[t] = np.max(Q[env.map[start_state]])

        s = next_s
        a = sample_random_action(Q,s)

    return Q, Q_initial_state

def plot(y,title,x_axis,y_axis, save_fig = False, fig_name = None ):

    plt.figure(figsize=(9.6,6.4))
    plt.plot(y)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.show()

    if save_fig:
        plt.savefig(fig_name)


if __name__ == '__main__':
    city = np.array([
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])


    # Create an environment maze
    env = City(city)
    start_state = (0,0,3,3);

    # Q-learning #
    gamma = 0.1
    epsilon = 0.3
    no_iterations = 10000000
    Q, Q_initial_state = Q_learning(env, gamma, epsilon, no_iterations, start_state)

    title = "Value Function over time for the initial state (0,0,3,3)"
    x_axis = 'iteration'
    y_axis = "Value"
    plot(Q_initial_state,title,x_axis,y_axis)#, save_fig = False, fig_name = None)

    # SARSA #
    eps = 0.1
    gamma = 0.1
    no_iterations = 10000000
    # bar since the best eps-greedy policy
    Q_bar, Q_initial_state = SARSA(env, eps, gamma, start_state, no_iterations)
    #title = "state value Function over time for the initial state (0,0,3,3)"
    #x_axis = 'iteration'
    #y_axis = "Value"
    #plot(Q_initial_state,title,x_axis,y_axis)#, save_fig = False, fig_name = None)
