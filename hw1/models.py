#!python3
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import floor
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

class CNN_LSTMRecognizer(nn.Module):
    def __init__(self,  input_dim,  output_dim,  hidden_dim, lstm_n_layers=1, bidirectional = False, dropout_rate=0, 
                    out_channels=6, kernel_size=6, stride=1, pooling_size=2, padding=0, dilation=1,):

        super(CNN_LSTMRecognizer, self).__init__()

        ##basic params
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        #cnn params
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling_size = pooling_size
        self.padding = padding
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels=1, out_channels= out_channels, kernel_size=kernel_size, 
                                                stride= stride, padding=padding, dilation=dilation)
        self.max_pool = nn.MaxPool1d(pooling_size, stride=stride)
        self.after_conv_dim = floor((input_dim+2*padding-dilation*(kernel_size-1)-1)/stride+1)
        self.after_pool_dim = floor((self.after_conv_dim+2*padding-dilation*(pooling_size-1)-1)/stride+1)
        # self.conv2lstm = nn.Linear()

        #lstm params
        self.lstm_n_layers = lstm_n_layers
        self.direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(self.after_pool_dim*out_channels, hidden_dim, num_layers=lstm_n_layers, bidirectional=bidirectional, dropout = dropout_rate)

        # fc layers
        self.h2h = nn.Linear(hidden_dim*self.direction, hidden_dim)
        self.h_dropout = nn.Dropout(p= dropout_rate)
        self.hidden2frame = nn.Linear(hidden_dim, output_dim)
        self.LogSoftmax = nn.LogSoftmax()
        

    def initHidden(self, n_sample):
        result = Variable(torch.zeros(self.lstm_n_layers * self.direction, n_sample, self.hidden_dim))
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

        input_conved = []
        for i in range(seq_len):
            single = input_[i,:,:] 
            #input of conv layer must be Input: (N,C,L)
            single = self.conv(single.unsqueeze(1)) 
            single = F.relu(single)
            single = self.max_pool(single)

            #resize
            dim_len = single.size()[1] * single.size()[2]
            single = single.view(n_sample, dim_len).unsqueeze(0)

            input_conved.append(single)
        # import pdb;pdb.set_trace()
        input_conved = torch.cat(input_conved, 0)

        output = input_conved
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

#TODO:
#CNN+dropout
