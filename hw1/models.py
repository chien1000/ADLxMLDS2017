#!python3
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
GPUID = 0

class LSTMRecognizer(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, bidirectional = False, dropout_rate=0):
        super(LSTMRecognizer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.direction = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout = dropout_rate)

        # The linear layer that maps from hidden state space to tag space
        self.h2h = nn.Linear(hidden_dim*self.direction, hidden_dim)
        self.h_dropout = nn.Dropout(p= dropout_rate)
        self.hidden2frame = nn.Linear(hidden_dim, output_dim)
        self.LogSoftmax = nn.LogSoftmax()

    #@staticmethod
    def initHidden(self, n_sample):
        result = Variable(torch.zeros(self.n_layers * self.direction, n_sample, self.hidden_dim))
        if USE_CUDA:
            return result.cuda(GPUID)
        else:
            return result

    
    def forward(self, input_, hidden=None, cell=None):
        if hidden is None:
            hidden = self.initHidden(input_.size()[1])
        if cell is None:
            cell = self.initHidden(input_.size()[1])
        
        seq_len = input_.size()[0]
        n_sample = input_.size()[1]

        # import pdb; pdb.set_trace()
        output = input_
        # for _ in range(self.n_layers):
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = F.relu(output)
        
        results = []
        for i in range(seq_len):
            h_output = self.h2h(output[i,:,:])
            h_output = F.relu(h_output)
            h_output = self.h_dropout(h_output)
            dist = self.LogSoftmax(self.hidden2frame(h_output))  
            results.append(dist)
        return results 
