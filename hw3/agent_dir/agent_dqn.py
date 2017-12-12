from agent_dir.agent import Agent
from agent_dir.model_dqn import Transition, ReplayMemory, DQN
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import math
from itertools import count
import pickle
import json
import time
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BATCH_SIZE = 32 #128
GAMMA = 0.99
TARGET_UPDATE_FREQ = 100
EVAL_UPDATE_FREQ = 4

NUM_EPISODES = 40000
ENV_STEPS = 10000000
LEARNING_START =  10000
MEMORY_SIZE = 10000

DECAY_STEPS  = ENV_STEPS / 10
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY =  DECAY_STEPS / 10 #200

PRINT_EVERY = 30
SAVE_EVERY = 1000
LEARNING_RATE = 0.01
SAVE_PREFIX = '2nets_mse'
SAVE_PATH = 'models'

PARAMS = {'BATCH_SIZE':BATCH_SIZE, 'GAMMA':GAMMA, 
        'TARGET_UPDATE_FREQ':TARGET_UPDATE_FREQ, 'EVAL_UPDATE_FREQ':EVAL_UPDATE_FREQ,  
          'NUM_EPISODES':NUM_EPISODES,  'ENV_STEPS':ENV_STEPS, 'DECAY_STEPS':DECAY_STEPS,
          'EPS_START': EPS_START, 'EPS_END':EPS_END, 'EPS_DECAY':EPS_DECAY, 'LEARNING_RATE':LEARNING_RATE,
          'LEARNING_START':LEARNING_START,'MEMORY_SIZE':MEMORY_SIZE}

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

plt.ion()

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)
        self.action_count = env.get_action_space().n

        self.eval_model = DQN(self.action_count)
        self.target_model = DQN(self.action_count)
        if use_cuda:
            self.eval_model.cuda()
            self.target_model.cuda()
        

        if args.test_dqn:
            #you can load your model here
            self.model_path = args.model_path
            self.eval_model.load_state_dict(torch.load(self.model_path, map_location={'cuda:0': 'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu'}))
            if use_cuda:
                self.eval_model.cuda()
            print('loading trained model')

        if args.train_dqn:
            self.optimizer = optim.RMSprop(self.eval_model.parameters(), lr=LEARNING_RATE)
            self.memory = ReplayMemory(MEMORY_SIZE)

            self.steps_done = 0
            self.update_done = 0
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
        pass

    def prepro_observation(self, observation):
        observation = torch.from_numpy(observation.transpose( (2, 0, 1)))
        observation = observation.unsqueeze(0)

        if use_cuda:
            observation = observation.cuda()
        return observation

    def train(self):
        # import pdb; pdb.set_trace()
        for i_episode in count(1): #range(1, NUM_EPISODES+1):
            print('Episode {}/{}, Step {}/{}'.format(i_episode, NUM_EPISODES, self.steps_done, ENV_STEPS))
            # Initialize the environment and state
            q_reward = 0.0
            observation = self.env.reset()
            observation = self.prepro_observation(observation)
            for t in count():
                # Select and perform an action
                action = self.make_action(observation, test=False)
                observation_next, reward, done, info = self.env.step(action[0, 0])
                observation_next = self.prepro_observation(observation_next)
                ## clip rewards between -1 and 1
                reward = max(-1.0, min(reward, 1.0))
                q_reward += reward
                reward = Tensor([reward])

                if done:
                    observation_next = None

                # Store the transition in memory
                self.memory.push(observation, action, observation_next, reward)

                # Move to the next state
                observation = observation_next
                self.steps_done += 1

                # Perform one step of the optimization (on the target network)
                if self.steps_done > LEARNING_START and self.update_done % EVAL_UPDATE_FREQ == 0:
                    self.optimize_model()
                if done:
                    self.episode_rewards.append(q_reward)
                    if i_episode % PRINT_EVERY == 0:
                        # self.plot_rewards()
                        print(self.episode_rewards[-29:])
                    break

            if i_episode % SAVE_EVERY ==0:
                try:

                    # pickle.dump((self.episode_rewards), open(os.path.join(self.save_path,'rewards.pkl'), 'wb'))
                    with open(os.path.join(self.save_path,'rewards.txt'), 'w') as f:
                        f.write('\n'.join(list(map(str,self.episode_rewards))))
                    torch.save(self.eval_model.state_dict(), open(os.path.join(self.save_path, 'epoch_{}_model.bin'.format(i_episode)), 'wb'))
                    print('--- save model ---')
                except Exception as e:
                    print('error occurs when saving data')
                    print(str(e))

            if self.steps_done > ENV_STEPS:
                break

        print('Complete')
        self.env.env.render(close=True)
        self.env.env.close()
        plt.ioff()
        plt.show()


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        if test:
            sample = random.random()
            if sample > 0.01:
                observation = self.prepro_observation(observation)
                # return self.model(Variable(observation, volatile=True).type(FloatTensor)).data.max(1)[1][0] #[1]: index matrix
                return self.eval_model.forward(Variable(observation,  volatile=True).type(FloatTensor)).data.max(1)[1][0] #[1]: index matrix
            else:
                return random.randrange(self.action_count)

        else:
            if self.steps_done < LEARNING_START:
                return LongTensor([[random.randrange(self.action_count)]])

            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
            # self.steps_done += 1
            # print('steps {}, eps_threshold {}'.format(self.steps_done, eps_threshold))
            if sample > eps_threshold:
                # return self.model(Variable(observation, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
                return self.eval_model.forward(Variable(observation, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
            else:
                return LongTensor([[random.randrange(self.action_count)]])


    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        if self.update_done % TARGET_UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())
        # import pdb; pdb.set_trace()
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        # non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]))
                                        
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.eval_model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
        # next_state_predict_values = self.target_model(non_final_next_states).detach()
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).detach().max(1)[0]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        # #next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.model.parameters():
            # if param.grad is not None:
                # param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.update_done +=1


    def plot_rewards(self):
        plt.figure(2)
        plt.clf()
        rewards_t = torch.FloatTensor(self.episode_rewards)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        # plt.plot(rewards_t.numpy())
        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 30:
            means = rewards_t.unfold(0, 30, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(29), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
