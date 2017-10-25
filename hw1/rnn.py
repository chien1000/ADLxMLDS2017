
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

# In[30]:


#from tqdm import tqdm 
from collections import defaultdict
# from itertools import chain
import random
import time 
import math
import pickle
import os
import json

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


def get_variable_from_seq(seq, seq_type, max_length=None):
    if max_length is not None:
        new_seq = pad_seq(seq, max_length, seq_type)
    else:
        new_seq = seq

    new_seq = np.array(new_seq)
    new_seq = torch.from_numpy(new_seq)
    if seq_type == 'feature' or seq_type =='f':
        new_seq = new_seq.float()
    new_seq = Variable(new_seq)
    
    return new_seq

def pad_seq(sequence, max_length, seq_type):
    diff = max_length - len(sequence)    
    if seq_type == 'features' or seq_type == 'f':
        dim = len(sequence[0])
        zero_v = np.zeros(dim)
        new_seq = sequence
        for _ in range(diff):
            new_seq.append(zero_v)

    elif seq_type == 'labels' or seq_type == 'l':
        new_seq = sequence + [PAD_IDX] * diff

    return new_seq

def batchify(fsequences, lsequences, shuffle=True):
    print('Creating batches')
    batches = defaultdict(list)
    processed_batches = []

    sent_uids = list(fsequences.keys())
    if shuffle:
        random.shuffle(sent_uids)

    for sent_uid in sent_uids:
        fseq = fsequences[sent_uid]
        lseq = lsequences[sent_uid]

        seq_len = len(lseq) 
        for b in BUCKETS:
            use_bucket = b
            if seq_len < b:
                break

        # fseq = pad_seq(fseq, use_bucket, 'f')
        # lseq = pad_seq(lseq, use_bucket, 'l')

        fseq_var = get_variable_from_seq(fseq, 'f', use_bucket)
        lseq_var = get_variable_from_seq(lseq, 'l', use_bucket)

        batches[use_bucket].append((fseq_var, lseq_var))

    for batch in batches.values():
#         print(len(batch))
        fbatch = [pair[0].unsqueeze(1) for pair in batch]
        lbatch = [pair[1].unsqueeze(1) for pair in batch]
        fbatch = torch.cat(fbatch, 1) #seq, batch, feature_dim
        lbatch = torch.cat(lbatch, 1)

        for pair in zip(fbatch.split(BATCH_LENGTH), lbatch.split(BATCH_LENGTH)):
            processed_batches.append(pair)

    #     torch.transpose(fbatch, 0, 1)
    #     torch.transpose(lbatch, 0, 1)
#         print(fbatch.size())
#         print(lbatch.size())
    return processed_batches


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

def trainEpochs(lstm, fsequences, lsequences, learning_rate, n_epochs, print_every, test_every, save_every, save_path, test_pairs=None):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    optimizer = optim.Adagrad(lstm.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(lstm.parameters(), lr=learning_rate, momentum=0.1)
    criterion = nn.NLLLoss()
    
    batches = batchify(fsequences, lsequences)
    n_batches = len(batches)
    print('total number of batch: %d'%(n_batches))
  
    for epoch in range(1, n_epochs+1):
        # set model for training
        lstm.train()

        print('epoch:{}/{}'.format(epoch, n_epochs))
        b = 1
        for fbatch, lbatch in batches:
            # print('[%d/%d]'%(b,n_batches),end='\r')
            b += 1
            input_variable = fbatch.cuda(GPUID) if USE_CUDA else fbatch
            target_variable = lbatch.cuda(GPUID) if USE_CUDA else lbatch
            predicts, loss = train(input_variable, target_variable, lstm, optimizer, criterion)
            print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / (print_every * n_batches)
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))
            
            print('-------testing with training data----------')
            # pre_ans = zip([aframe[0,0] for aframe in predicts], target_variable.data[:,0].tolist())
            # print('predicts, ans')
            # print([(p,a) for p, a in pre_ans if p!=a])
            error_count = 0 
            total = 0
            for i in range(input_variable.size()[1]): #batch length 
                for j in range(input_variable.size()[0]): #seq length
                    if predicts[j][i,0] != target_variable.data[j,i]:
                        error_count +=1
                    total +=1
            print('error rate: {} / {} = {}'.format(error_count, total, error_count/total))
        

        if epoch % test_every == 0:
            if test_pairs:
                lstm.eval()
                max_len = max(list(map(len,  [pair[1] for pair in test_pairs])))
                print(max_len)

                features = [get_variable_from_seq(pair[0], 'f', max_len) for pair in test_pairs]
                labels = [get_variable_from_seq(pair[1], 'l', max_len) for pair in test_pairs]
                features = [f.unsqueeze(1) for f in features]
                labels = [lab.unsqueeze(1) for lab in labels]
                features = torch.cat(features, 1) #seq, batch, feature_dim
                labels = torch.cat(labels, 1)
                features = features.cuda(GPUID) if USE_CUDA else features
                labels = labels.cuda(GPUID) if USE_CUDA else labels

                print('-------testing with testing data----------')
                results = lstm(features)
                predicts = []
                for i in range(features.size()[0]): #sequence length
                    dist = results[i]
                    topv, topi = dist.data.topk(1, 1)
                    predicts.append(topi)
                
                error_count = 0 
                total = 0
                for i in range(features.size()[1]): #batch length 
                    for j in range(features.size()[0]): #seq length
                        if predicts[j][i,0] != labels.data[j,i]:
                            error_count +=1
                        total +=1
                print('error rate: {} / {} = {}'.format(error_count, total, error_count/total))
            
        if epoch % save_every ==0:
            # set model for evaluate
            lstm.eval()
            torch.save(lstm.state_dict(), open(os.path.join(save_path, 'epoch_{}_model.bin'.format(epoch)), 'wb'))
            print('--- save model ---')


# In[50]:


USE_CUDA = torch.cuda.is_available()
GPUID = 0
HIDDEN_DIM = 256
N_LAYER = 2
BIDIRECTIONAL = True
DROPOUT_RATE = 0.4
BUCKETS = [10, 50, 100, 150, 200, 250 ,300, 350, 400, 450, 500, 550, 600, 650, 750, 800 ]
BATCH_LENGTH  = 100
PAD_TOKEN = 'PAD'
PAD_IDX = 0

SAVE_PREFIX = 'dropout'
NUM_EPOCH = 1500
PRINT_EVERY = 5
TEST_EVERY = 10
SAVE_EVERY = 20
LEARNING_RATE = 0.01
TESTING_NUM = 100
PARAMS = {'GPUID':GPUID, 'HIDDEN_DIM':HIDDEN_DIM, 'N_LAYER':N_LAYER, 'BIDIRECTIONAL':BIDIRECTIONAL,
            'DROPOUT_RATE':DROPOUT_RATE, 'NUM_EPOCH':NUM_EPOCH, 
            'LEARNING_RATE':LEARNING_RATE, 'BATCH_LENGTH': BATCH_LENGTH, 'PAD_IDX':PAD_IDX}

DATA_PATH = 'data'
SAVE_PATH = 'models'


# In[48]:


def main():

    data = read_ark('data/fbank/train.ark')
    labels = read_label_data('data/train.lab')

    idx_to_label = [PAD_TOKEN]
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


    train_fseq, test_fseq = {}, {}
    train_lseq, test_lseq = {}, {}
    shuffled_keys = list(fsequences.keys())
    random.shuffle(shuffled_keys)
    for i, skey in enumerate(shuffled_keys):
        if i < TESTING_NUM:
            test_fseq[skey] = fsequences[skey]
            test_lseq[skey] = lsequences[skey]
        else:
            train_fseq[skey] = fsequences[skey]
            train_lseq [skey] = lsequences[skey]
    print("training num = {} {} testing num ={} {}".format(len(train_fseq), len(train_lseq), len(test_fseq), len(test_lseq)))

    input_dim = len(list(fsequences.values())[0][0])
    output_dim = len(idx_to_label)
    PARAMS['input_dim'] = input_dim
    PARAMS['output_dim'] = output_dim
    
    lstm = LSTMRecognizer(input_dim = input_dim, 
                          hidden_dim = HIDDEN_DIM, 
                          output_dim = output_dim, 
                          n_layers = N_LAYER,
                          bidirectional = BIDIRECTIONAL,
                          dropout_rate = DROPOUT_RATE)

    print(lstm)

    if USE_CUDA:
        print('using CUDA models')
        lstm = lstm.cuda(GPUID)

    tmpf= {}
    tmpl= {}
    for i, key in enumerate(list(fsequences.keys())):
        tmpf[key] = fsequences[key]
        tmpl[key] = lsequences[key]
        if i > 5:
            break

    # save models and data to file
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    ts = "%d"%(time.time())
    params = '{}_HS_{}_EP_{}_LR_{}_NLAYER_{}_BIDIR_{}_DROPOUT_{}'.format(SAVE_PREFIX, HIDDEN_DIM, NUM_EPOCH, LEARNING_RATE, N_LAYER, int(BIDIRECTIONAL), DROPOUT_RATE)
    sub_path = os.path.join(SAVE_PATH,params)
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    else:
        sub_path = sub_path+ '_' + ts
        os.makedirs(sub_path)
    pickle.dump(idx_to_label, open(os.path.join(sub_path,'labels.pkl'), 'wb'))
    json.dump(PARAMS, open(os.path.join(sub_path, 'params.json'),'w'))

    # main training process
    plot_losses = trainEpochs(lstm = lstm, 
                              fsequences = train_fseq, 
                              lsequences = train_lseq, 
                              learning_rate = LEARNING_RATE, 
                              n_epochs = NUM_EPOCH, 
                              print_every = PRINT_EVERY, 
                              save_every = SAVE_EVERY, 
                              test_every = TEST_EVERY,
                              save_path = sub_path,
                              test_pairs = [ (test_fseq[suid], test_lseq[suid]) for suid in test_fseq])

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

