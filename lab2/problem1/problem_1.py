
# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 0 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 17th October 2020, by alessior@kth.se

### IMPORT PACKAGES ###
# numpy for numerical/random operations
# gym for the Reinforcement Learning environment
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

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


### Experience class ###

# namedtuple is used to create a special type of tuple object. Namedtuples
# always have a specific name (like a class) and specific fields.
# In this case I will create a namedtuple 'Experience',
# with fields: state, action, reward,  next_state, done.
# Usage: for some given variables s, a, r, s, d you can write for example
# exp = Experience(s, a, r, s, d). Then you can access the reward
# field by  typing exp.reward
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """
    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)

def fillExperienceBuffer( L, buffer, randomAgent):
    for e in range(L):

        state = env.reset()                    # Reset environment, returns
                                               # initial state
        done = False                           # Boolean variable used to indicate

        total_episode_reward = 0.
        t = 0
        while not done:

            # Compute output of the network
            action = randomAgent.forward(state)

            # The next line takes permits you to take an action in the RL environment
            # env.step(action) returns 4 variables:
            # (1) next state; (2) reward; (3) done variable; (4) additional stuff
            next_state, reward, done, _ = env.step(action)

            total_episode_reward += reward

            # Append experience to the buffer
            exp = Experience(state, action, reward, next_state, done)
            buffer.append(exp)

            # Update state for next iteration
            state = next_state
            t+= 1
    return buffer

### Neural Network ###
class MyNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, no_states, no_actions, hidden_layer_1_size = 64 , output_size = 10 ):
        super().__init__()

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(no_states, hidden_layer_1_size, bias= True)
        self.activation = nn.ReLU()

        #hidden_layer_1_size = 50
        hidden_layer_1_output_size = 64

        # Create input layer with ReLU activation
        self.hidden_layer_1 = nn.Linear(hidden_layer_1_size, hidden_layer_1_output_size, bias= True)

        hidden_layer_2_output_size = 64
        # Create input layer with ReLU activation
        self.hidden_layer_2 = nn.Linear(hidden_layer_1_output_size, hidden_layer_2_output_size, bias= True)

        # Create output layer
        self.output_layer = nn.Linear(hidden_layer_2_output_size, no_actions, bias= True)

    def forward(self, states):
        # Function used to compute the forward pass

        # Compute first layer

        l1 = self.input_layer(states)
        l1 = self.activation(l1)

        h1 = self.hidden_layer_1(l1)
        h1 = self.activation(h1)

        h2 = self.hidden_layer_2(h1)
        h2 = self.activation(h2)

        # Compute output layer
        out = self.output_layer(h2)
        return out




def computeEps(k, eps_max, eps_min, N_episodes, per = 0.95, linear = True):
    if linear:
        Z = per*N_episodes
        eps_k = max(eps_min, ((eps_max-eps_min)*(k-1))/(Z-1) )
        return eps_k
    else:
        Z = per*N_episodes
        eps_k = max(eps_min, eps_max*((eps_min)/(eps_max))**((k-1)/(Z-1)))
        return eps_k

def epsGreedyAction(k, eps_max, eps_min, N_episodes, action_distribution):

    eps_k = computeEps(k, eps_max, eps_min, N_episodes, 0.95, False)
    u = np.random.uniform(0, 1)

    if u < eps_k:
        # Explore
        return random.sample(range(0,len(action_distribution)), 1)[0]
    else:
        # Exploit
        return action_distribution.max(1)[1].item()


def computeTarget(states, actions, rewards, next_states, dones, Q_theta_p, gamma):

    y = []
    batch_size = len(states)

    for experience in range(batch_size):

        s = states[experience]
        a = actions[experience]
        r = rewards[experience]
        next_s = next_states[experience]
        done = dones[experience]

        if done:
            y.append(r)
        else:
            action_distribution = Q_theta_p(torch.tensor(next_s,requires_grad=False))

            Q_theta_p_max = float(max(action_distribution))
            y.append(r + gamma * Q_theta_p_max)
    return y


### CREATE RL ENVIRONMENT ###
# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

def main():

    # Parameters
    N_episodes = 500                             # Number of episodes
    gamma = 0.99                      # Value of the discount factor
    n_ep_running_average = 50                    # Running average of 50 episodes
    n_actions = env.action_space.n               # Number of available actions
    dim_state = len(env.observation_space.high)  # State dimensionality


    eps_max = 0.99
    eps_min = 0.05

    C = 0

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode

    # Random agent initialization
    randomAgent = RandomAgent(n_actions)

    L=20000 #5000-30000
    
    ### Create Experience replay buffer ###
    buffer = ExperienceReplayBuffer(maximum_length=L)

    TE = trange(N_episodes, desc='Episode: ', leave=True)

    ### Create network ###
    #network = MyNetwork(input_size=n, output_size=m)
 
    N = 32 #4-128
    C = L/N

    buffer = fillExperienceBuffer(32, buffer, randomAgent )
    #torch.save(buffer)


    Q_theta = MyNetwork(dim_state, n_actions)
    Q_theta_p = MyNetwork(dim_state, n_actions)

    Q_theta_best = None
    reward_avg_max = -1000000


    ### Create optimizer ###
    optimizer = optim.Adam(Q_theta.parameters(), lr=0.0005) # 10-3 and 10^-4



    ### TRAINING ###
    # Perform training only if we have more than 3 elements in the buffer
    training_steps=0
    
    for k in TE:
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0

        while not done:

            state_tensor = torch.tensor([state],
                                        requires_grad=False,
                                        dtype=torch.float32)

            action_distribution = Q_theta(state_tensor)
            action = epsGreedyAction(k, eps_max, eps_min, N_episodes, action_distribution)

            next_state, reward, done, _ = env.step(action)

            exp = Experience(state, action, reward, next_state, done)
            buffer.append(exp)

            # Sample a batch of 3 elements
            states, actions, rewards, next_states, dones = buffer.sample_batch(n=N)

            # TD updated values
            y = computeTarget(states, actions, rewards, next_states, dones, Q_theta_p, gamma)
            y_tensor=torch.tensor(y, requires_grad=False, dtype=torch.float32)

            # Training process, set gradients to 0
            optimizer.zero_grad()

            # Compute output of the network given the states batch
            batch_action_values = Q_theta(torch.tensor(states,
                            requires_grad=False,
                            dtype=torch.float32))

            index_tensor = torch.tensor(np.arange(0, N, 1))
            actions_tensor = torch.tensor(actions, requires_grad=False, dtype=torch.int64)
            
            #action_values = [batch_action_values[i][actions[i]] for i in range (batch_action_values.shape[0]) ]
            
            action_values=batch_action_values[index_tensor, actions_tensor]
            #action_values_tensor=torch.tensor(action_values, requires_grad=False, dtype=torch.float32)

            # Compute loss function
            #pdb.set_trace()
            loss = nn.functional.mse_loss(action_values, y_tensor)

            # Compute gradient
            loss.backward()

            # Clip gradient norm to 1
            nn.utils.clip_grad_norm_(Q_theta.parameters(), max_norm=1.) # between 0.5 and 2

            # Perform backward pass (backpropagation)
            optimizer.step()
            training_steps+=1
            
            if np.mod(training_steps, C) == 0:
                Q_theta_p = Q_theta
                #pdb.set_trace()

            t+= 1
            total_episode_reward +=reward
            state = next_state


        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)
        if k>50:
            reward_avg = running_average(episode_reward_list, 50)[-1]
            print ("Current reward avg", reward_avg, "episode:", k, "number of steps", t)
            if reward_avg > reward_avg_max:
                print("Best running avg so far: ", reward_avg, "episode:", k)
                reward_avg_max = reward_avg
                Q_theta_best = Q_theta
            if reward_avg>100:
                Q_theta_best=Q_theta
                break

    torch.save(Q_theta_best, 'neural-network-1.pth')
    #print(episode_reward_list)
    #print(episode_number_of_steps)

    
    # Close all the windows
    env.close()
    
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, len(episode_reward_list)+1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    
    ax[1].plot([i for i in range(1, len(episode_reward_list)+1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, len(episode_reward_list)+1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()
    
if __name__ == "__main__":
    main()