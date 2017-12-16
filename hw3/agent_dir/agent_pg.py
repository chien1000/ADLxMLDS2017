from agent_dir.agent import Agent
from agent_dir.model_pg import Policy
import scipy
import scipy.misc
import os
import json
import time
import numpy as np
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

TOTAL_EPISODE = 10000
LEARNING_RATE = 0.0001
GAMMA = 0.99
BASELINE = 1
PRINT_EVERY = 1
SAVE_EVERY = 100
SAVE_PREFIX = 'BASELINE'
SAVE_PATH = 'pg_models'


PARAMS = {'LEARNING_RATE':LEARNING_RATE, 'GAMMA':GAMMA, 
        'BASELINE':BASELINE}
        

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
# dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
# class Variable(torch.autograd.Variable):
#     def __init__(self, data, *args, **kwargs):
#         if use_cuda:
#             data = data.cuda()
#         super(Variable, self).__init__(data, *args, **kwargs)



def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)

def state_to_variable(state, to_prepro=True):
    if to_prepro:
        state = prepro(state)

    state = state.transpose(2, 0, 1).astype(float)
    var = Variable(torch.FloatTensor(state))
    var = var.unsqueeze(0)
    if use_cuda:
        var = var.cuda()

    return var

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        self.args =args 
        self.env = env
        self.action_count = env.get_action_space().n
        self.policy = Policy(self.action_count)
        if use_cuda:
            self.policy.cuda()

        if args.test_pg:
            random.seed(136)
            self.model_path = args.model_path or 'models/pg/model.bin'
            self.policy.load_state_dict(torch.load(self.model_path, map_location={'cuda:0': 'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu'}))
            if use_cuda:
                self.policy.cuda()
            print('loading trained model')
            self.is_first =True
            self.last_state = None

        elif args.train_pg:
            
            self.optimizer = optim.RMSprop(self.policy.parameters(), lr= LEARNING_RATE)
            self.episode_rewards = []

            params = '{}_LR_{}_GAMMA_{}'.format(SAVE_PREFIX, LEARNING_RATE, GAMMA)
            self.save_path = os.path.join(SAVE_PATH,params)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            else:
                ts = "%d"%(time.time())
                self.save_path = self.save_path+ '_' + ts
                os.makedirs(self.save_path)
            json.dump(PARAMS, open(os.path.join(self.save_path, 'params.json'),'w'))

        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.is_first = True

    # def select_action(self, state):
    #     state = torch.from_numpy(state).type(dtype).unsqueeze(0)
    #     probs = self.policy(Variable(state))
    #     m = Categorical(probs)
    #     action = m.sample()
    #     return action.data[0], m.log_prob(action)

    def optimize(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.policy.rewards[::-1]:
            if r==1 or r==-1:
                R = 0 
            R = r + GAMMA * R
            # R -= BASELINE
            rewards.insert(0, R)
        # print(self.policy.rewards)
        # print(rewards)
        rewards = Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        rewards = rewards - BASELINE
        rewards = Variable(rewards, requires_grad=False)
        # import pdb;pdb.set_trace()
        saved_log_probs = torch.cat(self.policy.saved_log_probs)
        policy_loss = (rewards * saved_log_probs*(-1)).sum()

        # for log_prob, reward in zip(self.policy.saved_log_probs, rewards):
        #     policy_loss.append(-log_prob * reward)
        # policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def train(self):
        for i_episode in count(1):
            print('Episode {}'.format(i_episode))
            # import pdb; pdb.set_trace()
            state = self.env.reset()
            last_state = state
            for t in range(self.env.env.spec.timestep_limit):  # Don't infinite loop while learning
                state_diff = state - last_state
                action, log_prob = self.make_action(state_diff, test = False)

                last_state = state
                state, reward, done, _ = self.env.step(action)

                self.policy.rewards.append(reward)
                self.policy.saved_log_probs.append(log_prob)

                if done:
                    break

            self.episode_rewards.append(sum(self.policy.rewards))
            self.optimize()
            if i_episode % PRINT_EVERY == 0:
                print(self.episode_rewards[-PRINT_EVERY:])
            
            if i_episode % SAVE_EVERY ==0:
                try:

                    # pickle.dump((self.episode_rewards), open(os.path.join(self.save_path,'rewards.pkl'), 'wb'))
                    with open(os.path.join(self.save_path,'rewards.txt'), 'w') as f:
                        f.write('\n'.join(list(map(str,self.episode_rewards))))
                    torch.save(self.policy.state_dict(), open(os.path.join(self.save_path, 'epoch_{}_model.bin'.format(i_episode)), 'wb'))
                    print('--- save model ---')
                except Exception as e:
                    print('error occurs when saving data')
                    print(str(e))

            if i_episode >= TOTAL_EPISODE:
                break

        

        
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """

        if test:
            if self.is_first:
                state_diff = observation - observation
            else:
                state_diff = observation - self.last_state

            state_var = state_to_variable(state_diff)
            # a = self.policy(state_var).data.max(1)[1][0]
            probs = self.policy(state_var)
            m = Categorical(probs)
            action = m.sample()
            a = action.data[0]
            self.last_state = observation

            return a
        else:
            state_var = state_to_variable(observation)
            probs = self.policy(state_var)
            m = Categorical(probs)
            action = m.sample()
            
            return action.data[0], m.log_prob(action)

