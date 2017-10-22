
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable


# In[30]:


from tqdm import tqdm 
from collections import defaultdict
# from itertools import chain
import random
import time 
import math
import pickle
import os
import json


# In[3]:


SOS_TOKEN = 'SOS'
EOS_TOKEN = 'EOS'
SOS_IDX = 0
EOS_IDS = 1
SOS_VECTOR = None
EOS_VECTOR = None
# BUCKETS = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]


# In[4]:


def read_ark(file_name):
    data = {}
    with open(file_name,'r') as fin:
        for line in fin.readlines():
            line= line.strip()
    #         print(line)
            elements = line.split(' ')
            features = elements[1:]
            iid = elements[0]

            iid_elements = iid.split('_')
            speaker_id = iid_elements[0]
            gender = speaker_id[0]
            sent_id = iid_elements[1]
            frame_id = iid_elements[2]

            data[(speaker_id, sent_id, frame_id)] = { 'speaker_id': speaker_id, 'gender': gender, 'features': features}

    #         print('{} {} {} {}'.format(speaker_id, gender, sent_id, frame_id))
    
#     print(len(data))
#     for k in data:
#         print(k)
#         print(data[k])
#         break
    
    return data


# In[5]:


def read_label_data(file_name):
    labels = {}
    with open(file_name, 'r') as fin:
        for line in fin.readlines():
            elements = line.strip().split(',')
            label = elements[1]
            iid = elements[0]

            iid_elements = iid.split('_')
            speaker_id = iid_elements[0]
            sent_id = iid_elements[1]
            frame_id = iid_elements[2]
            
            labels[(speaker_id, sent_id, frame_id)] = label
    return labels


# In[6]:


def make_training_data(data, labels, label_to_idx):
#     all_sent = set(k[0] for k in data.keys())
    sort_keys = sorted(data.keys(), key = lambda x: (x[0], x[1], int(x[2])))
#     print(sort_keys[:100])

    fsequences = defaultdict(list)
    lsequences = defaultdict(list)

    for iid in sort_keys:
        speaker_id = iid[0]
        sent_id = iid[1]

        info = data[iid]
        features = list(map(float, info['features']))
#         features = info['features']
        if info['gender'] == 'f':
            features.append(0)
        else:
            features.append(1)
    
        fsequences[(speaker_id, sent_id)].append(features)
        l = label_to_idx[labels[iid]]
        lsequences[(speaker_id, sent_id)].append(l)
                
    return fsequences, lsequences
    
# make_training_data(data, labels)


# In[20]:


def get_variable_from_seq(seq, seq_type):
    if seq_type == 'feature' or seq_type =='f':
#         tmpk = random.choice(list(data.keys()))
#         dim = len(data[k]['features'])
        dim = len(seq[0])
        global SOS_VECTOR
        global EOS_VECTOR
        if SOS_VECTOR is None:
            SOS_VECTOR = np.random.rand(dim)
        if EOS_VECTOR is None:
            EOS_VECTOR = np.random.rand(dim)
        
        new_seq = [SOS_VECTOR]
        new_seq.extend(seq)
        new_seq.append(EOS_VECTOR)
        
    elif seq_type == 'label' or seq_type == 'l':

        new_seq = [SOS_IDX]
        new_seq.extend(seq)
        new_seq.append(EOS_IDS)
    
    new_seq = np.array(new_seq)
    new_seq = torch.from_numpy(new_seq)
    if seq_type == 'feature' or seq_type =='f':
        new_seq = new_seq.float()
    new_seq = Variable(new_seq)
    
    return new_seq


# In[8]:


def batchify(fsequences, lsequences):
    print('Crating batches')
    batches = defaultdict(list)
    processed_batches = []

    for sent_uid in fsequences:
        fseq = fsequences[sent_uid]
        fseq_var = get_variable_from_seq(fseq, 'f')
        lseq = lsequences[sent_uid]
        lseq_var = get_variable_from_seq(lseq, 'l')
        seq_len = len(lseq) + 2
        batches[seq_len].append((fseq_var, lseq_var))

    print(len(batches))
    for batch in batches.values():
#         print(len(batch))
        fbatch = [pair[0].unsqueeze(1) for pair in batch]
        lbatch = [pair[1].unsqueeze(1) for pair in batch]
        fbatch = torch.cat(fbatch, 1)
        lbatch = torch.cat(lbatch, 1)

    #     torch.transpose(fbatch, 0, 1)
    #     torch.transpose(lbatch, 0, 1)
#         print(fbatch.size())
#         print(lbatch.size())

        processed_batches.append((fbatch, lbatch))

    return processed_batches


# In[9]:


USE_CUDA = torch.cuda.is_available()
GPUID = 0
HIDDEN_SIZE = 128
N_LAYER = 1


# In[10]:


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[11]:


class LSTMRecognizer(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(LSTMRecognizer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2frame = nn.Linear(hidden_dim, output_dim)
        self.LogSoftmax = nn.LogSoftmax()

    def initHidden(self, n_sample):
        result = Variable(torch.zeros(self.n_layers, n_sample, self.hidden_dim))
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

        output = input_
        for _ in range(self.n_layers):
            output, (hidden, cell) = self.lstm(output, (hidden, cell))
        
        results = []
        for i in range(seq_len):
            dist = self.LogSoftmax(self.hidden2frame(output[i,:,:]))
            results.append(dist)
        return results 


# In[45]:


def train(input_variable, target_variable, lstm, optimizer, criterion):
    loss = 0

    optimizer.zero_grad()

    n_sample = input_variable.size()[1]
    seq_len = input_variable.size()[0]
    # encoder forward
    results = lstm(input_variable)

    predict_idx = []
    for i in range(seq_len):
        dist = results[i]
        loss += criterion(dist, target_variable[i, :])
        topv, topi = dist.data.topk(1, 1)
        predict_idx.append(topi)

    loss.backward()
    optimizer.step()

    return predict_idx, loss.data[0] / seq_len

def trainEpochs(lstm, fsequences, lsequences, learning_rate, n_epochs, print_every, save_every, save_path, test_pairs=None):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    optimizer = optim.Adagrad(lstm.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    batches = batchify(fsequences, lsequences)
    n_batches = len(batches)
    print('total number of batch: %d'%(n_batches))
    
#     ts = "%d"%(time.time())
#     PN = 'BS-'+str(BATCH_LENGTH)+'_HS-'+str(HIDDEN_SIZE)+'_AM-'+str(ATTN_METHOD)\
#         +'_DR-'+str(DECODER_DROPOUT)+'_EP-'+str(NUM_EPOCH)+'_LR-'+str(LEARNING_RATE)

    for epoch in range(1, n_epochs+1):
        # set model for training
        lstm.train()

        print('epoch:{}/{}'.format(epoch, n_epochs))
        b = 1
        for fbatch, lbatch in batches:
            print('[%d/%d]'%(b,n_batches),end='\r')
            b += 1
            input_variable = fbatch.cuda(GPUID) if USE_CUDA else fbatch
            target_variable = lbatch.cuda(GPUID) if USE_CUDA else lbatch
            predicts, loss, = train(input_variable, target_variable, lstm, optimizer, criterion)
            print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / (print_every * n_batches)
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))
            
            print('-------testing with training data----------')
            pre_ans = zip([aframe[0,0] for aframe in predicts], target_variable.data[:,0].tolist())
            print('predicts, ans')
            print([(p,a) for p, a in pre_ans if p!=a])
            
            if test_pairs:
                features, labels = test_pairs[0], test_pairs[1]
                print('-------testing with testing data----------')
                results = lstm(features)
                predicts = []
                for i in range(features.size()[0]): #sequence length
                    dist = results[i]
                    topv, topi = dist.data.topk(1, 1)
                    predicts.append(topi)
                
                for i in range(features.size()[1]): #batch length 
                    print('~~~', [aframe[i,0] for aframe in predicts])
                    print('===', target_variable.data[:,i].tolist() )
            
        if epoch % save_every ==0:
            # set model for evaluate
            lstm.eval()
            torch.save(lstm.state_dict(), open(os.path.join(save_path, 'epoch_{}_model.bin'.format(epoch)), 'wb'))
            print('--- save model ---')


# In[50]:


NUM_EPOCH = 100
PRINT_EVERY = 5
SAVE_EVERY = 5
LEARNING_RATE = 0.01
TESTING_NUM = 5
PARAMS = {'SOS_TOKEN':SOS_TOKEN, 'EOS_TOKEN':EOS_TOKEN, SOS_IDX:'SOS_IDX', EOS_IDS:'EOS_IDS',
          'GPUID':GPUID, 'HIDDEN_SIZE':HIDDEN_SIZE, 'N_LAYER':N_LAYER,'NUM_EPOCH':NUM_EPOCH, 
          'LEARNING_RATE':LEARNING_RATE}
DATA_PATH = 'data'
SAVE_PATH = 'models'
SAVE_PREFIX = ''


# In[48]:


def main():

    data = read_ark('data/mfcc/train.ark')
    labels = read_label_data('data/train.lab')

    idx_to_label = [SOS_TOKEN, EOS_TOKEN]
    idx_to_label.extend(list(set(labels.values())))
    label_to_idx = {l:i for i, l in enumerate(idx_to_label)}

    print(len(data))
    for k in data:
        print(k)
        print(data[k])
        print(labels[k])
        break
        
    print(idx_to_label)
    print(label_to_idx)

    fsequences, lsequences = make_training_data(data, labels, label_to_idx)
    print('number of sentence: {}'.format(len(fsequences)))

    seq_len = list(map(len, fsequences.values()))
    # print(set(seq_len))
    print('max sequence length: {}'.format(max(seq_len)))
    print('min sequence length: {}'.format(min(seq_len)))
    print('average sequence length: {}'.format(sum(seq_len)/len(seq_len)))

    input_dim = len(list(fsequences.values())[0][0])
    output_dim = len(idx_to_label)
    PARAMS['input_dim'] = input_dim
    PARAMS['output_dim'] = output_dim
    
    lstm = LSTMRecognizer(input_dim = input_dim, 
                          hidden_dim = HIDDEN_SIZE, 
                          output_dim = output_dim, 
                          n_layers = N_LAYER)

    print(lstm)

    if USE_CUDA:
        print('using CUDA models')
        lstm = lstm.cuda(GPUID)

    # tmpf= {}
    # tmpl= {}
    # for i, key in enumerate(list(fsequences.keys())):
        # tmpf[key] = fsequences[key]
        # tmpl[key] = lsequences[key]
        # if i > 5:
            # break

    # save models and data to file
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    ts = "%d"%(time.time())
    params = '{}_HS_{}_EP_{}_LR_{}'.format(SAVE_PREFIX, HIDDEN_SIZE, NUM_EPOCH, LEARNING_RATE)
    sub_path = os.path.join(SAVE_PATH,params)
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    else:
        sub_path = sub_path+ '_' + ts
        os.makedirs(sub_path)
    pickle.dump((idx_to_label, SOS_VECTOR, EOS_VECTOR), open(os.path.join(sub_path,'data.pkl'), 'wb'))
    json.dump(PARAMS, open(os.path.join(sub_path, 'params.json'),'w'))

    # main training process
    plot_losses = trainEpochs(lstm = lstm, 
                              fsequences = fsequences, 
                              lsequences = lsequences, 
                              learning_rate = LEARNING_RATE, 
                              n_epochs = NUM_EPOCH, 
                              print_every = PRINT_EVERY, 
                              save_every = SAVE_EVERY, 
                              save_path = sub_path,
                              test_pairs=None)

    # set model for evaluation
    lstm.eval()

    # print('-------testing with training data----------')
    # evaluateRandomly(encoder, attn_decoder, input_lang_word, input_lang_sense, output_lang, pairs_word[:num_pairs-TESTING_NUM], pairs_sense[:num_pairs-TESTING_NUM], MAX_LENGTH, 1)
    # print('')
    # print('-------testing with testing data-----------')
    # evaluateRandomly(encoder, attn_decoder, input_lang_word, input_lang_sense, output_lang, pairs_word[num_pairs-TESTING_NUM:], pairs_sense[num_pairs-TESTING_NUM:], MAX_LENGTH, 1)

    torch.save(lstm.state_dict(), open(os.path.join(sub_path, 'model.bin'), 'wb'))
    print('All of models are trained and saved to {}'.format(sub_path))


# In[ ]:


# def evaluate(lstm, , max_length, sentence_word, sentence_sense):
#     input_variable_word = variableFromSentence(input_lang_word, sentence_word)
#     input_variable_sense = variableFromSentence(input_lang_sense, sentence_sense)
    
#     input_variable = torch.cat((input_variable_word, input_variable_sense), 1)

#     n_sample = input_variable.size()[0]
#     input_length = input_variable.size()[1] // 2

#     # encoder forward
#     input_variable = input_variable.cuda(GPUID) if USE_CUDA else input_variable
#     # encoder_hiddens, encoder_hidden, encoder_cell = encoder(input_variable)
#     encoder_outputs, encoder_hidden_word, encoder_hidden_sense, encoder_cell_word, encoder_cell_sense = encoder(input_variable)
    

#     # prepare decoder input data which the fist input is the index of start of sentence
#     decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]]))  # SOS
#     decoder_input = decoder_input.cuda(GPUID) if USE_CUDA else decoder_input
#     decoder_hidden = encoder_outputs[:, -1, :].unsqueeze(0)
#     #decoder_cell = encoder_cell
#     decoder_cell = decoder.initHidden(n_sample) # get cell with zero value

#     decoded_words = []
#     decoder_attentions = torch.zeros(max_length, input_length)
#     assert max_length > 0
#     for di in range(max_length):
#         decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
#             decoder_input, decoder_hidden, decoder_cell, encoder_outputs)

#         decoder_attentions[di] = decoder_attention.data
#         topv, topi = decoder_output.data.topk(1)
#         ni = topi[0][0]
#         if ni == EOS_TOKEN:
#             decoded_words.append('<EOS>')
#             break
#         else:
#             decoded_words.append(output_lang.index2word[ni])

#         decoder_input = Variable(torch.LongTensor([[ni]]))
#         decoder_input = decoder_input.cuda(GPUID) if USE_CUDA else decoder_input

#     return decoded_words, decoder_attentions[:di + 1]


# print(fsequences[('faem0', 'si1392')][:10])
# print(lsequences[('faem0', 'si1392')][:10])
# print(fsequences[('faem0', 'si1392')][-10:])
# print(lsequences[('faem0', 'si1392')][-10:])



if __name__ == '__main__':
    main()

