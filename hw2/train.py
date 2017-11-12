# coding: utf-8

from os import listdir
from os.path import isfile, join, basename
import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

from collections import defaultdict
import random
import time
import pickle
import math


PAD_IDX = 0
START_IDX = 1
END_IDX = 2
PAD_TOKEN = 'PAD'
START_TOKEN = 'START'
END_TOKEN = 'END'
INPUT_MAX_LENGTH = 80
TARGET_MAX_LENGTH = 40

HIDDEN_DIM = 256
N_LAYER = 1
BIDIRECTIONAL = False
DROPOUT_RATE = 0

USE_CUDA = torch.cuda.is_available()
BATCH_SIZE  = 128
NUM_EPOCH = 10
PRINT_EVERY = 1
TEST_EVERY = 1
SAVE_EVERY = 1
LEARNING_RATE = 0.01
TESTING_NUM = 1
SAVE_PREFIX = ''
# DATA_PATH = 'data'
SAVE_PATH = 'models'

PARAMS = {'HIDDEN_DIM':HIDDEN_DIM, 'N_LAYER':N_LAYER, 
        'BIDIRECTIONAL':BIDIRECTIONAL, 'DROPOUT_RATE':DROPOUT_RATE,  
          'NUM_EPOCH':NUM_EPOCH,  'LEARNING_RATE':LEARNING_RATE, 
          'BATCH_SIZE': BATCH_SIZE, 'PAD_IDX':PAD_IDX, 'START_IDX':START_IDX, 'END_IDX':END_IDX}

def read_training_data(data_path):
    feat_path = join(data_path,'training_data/feat')
    feat_files = [join(feat_path, f) for f in listdir(feat_path) if isfile(join(feat_path, f))]
    
    label_path = join(data_path, 'training_label.json')
    with open(label_path,'r') as fin:
        train_labels = json.load(fin)
    vid_to_caption = {item['id']:item['caption'] for item in train_labels}
    
    vfeats = []
    captions = []
    vids = []
    for feat_f in feat_files:
        vid = os.path.splitext(basename(feat_f))[0]
        cap = vid_to_caption[vid]
        feat = np.load(feat_f)

        vfeats.append(feat)
        captions.append(cap)
        vids.append(vid)
    
    return vids, vfeats, captions

def get_dictionary(sentence_lst):
    terms = set()
    for sent in sentence_lst:
        if "str" in str(type(sent)):
            ss = sent.split()
        else:
            ss = sent
#         ss =  [term.strip('.!,()').lower() for term in ss]
        terms.update(ss)
    
    terms.discard('')
    idx2term = [PAD_TOKEN, START_TOKEN, END_TOKEN] + list(terms)
    term2idx = {t:i for i, t in enumerate(idx2term)}
        
    return idx2term, term2idx

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


def get_variable_from_seq(seq, seq_type, max_length=None):
    if max_length is not None:
        new_seq = pad_seq(seq, max_length, seq_type)
    else:
        new_seq = seq
    
    if str(type(new_seq)) == '<class \'list\'>':
        new_seq = np.array(new_seq)
    new_seq = torch.from_numpy(new_seq)
    if seq_type == 'feature' or seq_type =='f':
        new_seq = new_seq.float()
    new_seq = Variable(new_seq)
    
    return new_seq


# # def make_training_pairs(vfeats, processed_captions, term2idx):
# input_vars = []
# output_vars = []
# for vfeat, multi_caps in zip(vfeats, processed_captions):
    
#     input_var = get_variable_from_seq(vfeat, 'f')
#     output_var = []
#     for cap in multi_caps:
#         cap_idx = [START_IDX] + [term2idx[term] for term in cap] + [END_IDX]
#         cap_var = get_variable_from_seq(cap_idx, 'l', 40)
#         output_var.append(cap_var)
    
    
#     input_vars.append(input_var)
#     output_vars.append(output_var)

class MyDataset(Dataset):
    def __init__(self, input_lst, target_lst):
        self.input_lst = input_lst
        self.target_lst = target_lst
        
        assert len(input_lst) == len(target_lst)


    def __getitem__(self, index):
        input_   = self.input_lst[index]
        target = self.target_lst[index]

        return input_, target


    def __len__(self):
        return len(self.input_lst)
    
# train_dataset = MyDataset(vfeats, chosen_captions)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def batchify(inputs, targets, shuffle=True):
    
    print('Creating batches')
    n_data = len(inputs)
    batches = []

    all_idx = list(range(len(inputs)))
    if shuffle:
        random.shuffle(all_idx)
        inputs = [inputs[idx] for idx in all_idx]
        targets = [targets[idx] for idx in all_idx]
    
    steps = list(range(0, n_data+1, BATCH_SIZE))
    if steps[-1] < n_data:
        steps.append(n_data)
    
    print(steps)
    for i in range(len(steps)-1):
        st = steps[i]
        ed = steps[i+1]
        batch_inputs = inputs[st:ed]
        batch_targets = targets[st:ed]
        
        batch_inputs = [get_variable_from_seq(input_,'f').unsqueeze(1) for input_ in batch_inputs]
        batch_targets = [get_variable_from_seq(target,'l').unsqueeze(1) for target in batch_targets]
        
        batch_inputs = torch.cat(batch_inputs, 1) #seq, batch, feature_dim
        batch_targets = torch.cat(batch_targets, 1)
        
        batches.append((batch_inputs, batch_targets))

    return batches


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=1):

    loss = 0

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    n_sample = input_variable.size()[1]
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # encoder forward
    #encoder_outputs: (seq_len, batch, hidden_size * num_directions)
    encoder_outputs, (encoder_hidden, encoder_cell) = encoder(input_variable) 

    # prepare decoder input data which the fist input is the index of start of sentence
    decoder_input = Variable(torch.LongTensor([[START_IDX]*n_sample])).view(-1, 1)
    decoder_input = decoder_input.cuda(GPUID) if USE_CUDA else decoder_input

    ##TODO choose what to initialize?
    decoder_hidden = encoder_outputs[-1, :, :].unsqueeze(0).contiguous()
    #decoder_cell = encoder_cell # this is using last hidden state for decoder inital hidden state
    decoder_cell = decoder.initHidden(n_sample) # get cell with zero value

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
            decoder_input, decoder_hidden, decoder_cell, encoder_outputs)

        loss += criterion(decoder_output, target_variable[di, :])
        use_teacher_forcing =False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_input = target_variable[di:di+1, :].view(-1, 1)
        else:
            # Without teacher forcing: use its own predictions as the next input
            topv, topi = decoder_output.data.topk(1)
            ni = topi
            decoder_input = Variable(ni).cuda(GPUID) if USE_CUDA else Variable(ni)

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def evaluate(encoder, decoder, input_variable, idx2term, max_length):

    n_sample = input_variable.size()[1]

    # encoder forward
    input_variable = input_variable.cuda(GPUID) if USE_CUDA else input_variable
    encoder_hiddens, (encoder_hidden, encoder_cell) = encoder(input_variable)

    # prepare decoder input data which the fist input is the index of start of sentence
    decoder_input = Variable(torch.LongTensor([[START_IDX]]))  # SOS
    decoder_input = decoder_input.cuda(GPUID) if USE_CUDA else decoder_input
    decoder_hidden = encoder_hiddens[-1, :, :].unsqueeze(0)
    #decoder_cell = encoder_cell
    decoder_cell = decoder.initHidden(n_sample) # get cell with zero value

    decoded_words = []

    assert max_length > 0
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
            decoder_input, decoder_hidden, decoder_cell, encoder_hiddens)

        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == END_IDX:
            decoded_words.append(END_TOKEN)
            break
        else:
            decoded_words.append(idx2term[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda(GPUID) if USE_CUDA else decoder_input

    return decoded_words


def trainEpochs(encoder, decoder, vfeats, processed_captions, learning_rate, n_epochs, idx2term, print_every, test_every, save_every, save_path, test_pairs=None):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.Adagrad(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adagrad(decoder.parameters(), lr=learning_rate*0.1)

    criterion = nn.NLLLoss()
  
    for epoch in range(1, n_epochs+1):
        print('epoch:{}/{}'.format(epoch, n_epochs))

        # set model for training
        encoder.train()
        decoder.train()
        
        chosen_captions = [random.choice(multi_caps) for multi_caps in processed_captions]
    
        batches = batchify(vfeats, chosen_captions)
        n_batches = len(batches)
        
        for b, (input_variable, targets) in enumerate(batches):
            print('[%d/%d]'%(b,n_batches),end='\r')
        
            if USE_CUDA:
                input_variable, targets = input_variable.cuda(), targets.cuda()

            loss = train(input_variable, targets, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=1)
            print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / (print_every * n_batches)
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))
            
            print('-------testing with training data----------')
            for i in range(3):
                test_var = input_variable[:, i, :].unsqueeze(1)
                pred_words = evaluate(encoder, decoder, test_var, idx2term, TARGET_MAX_LENGTH)
                print('===\n' + ' '.join([idx2term[tidx] for tidx in targets[:,i].data.tolist()]))
                print('>>>\n' + ' '. join(pred_words))
                print('')

#         if epoch % test_every == 0:
#             if test_pairs:
#                 lstm.eval()
#                 max_len = max(list(map(len,  [pair[1] for pair in test_pairs])))

#                 features = [get_variable_from_seq(pair[0], 'f', max_len) for pair in test_pairs]
#                 labels = [get_variable_from_seq(pair[1], 'l', max_len) for pair in test_pairs]
#                 features = [f.unsqueeze(1) for f in features]
#                 labels = [lab.unsqueeze(1) for lab in labels]
#                 features = torch.cat(features, 1) #seq, batch, feature_dim
#                 labels = torch.cat(labels, 1)
#                 features = features.cuda(GPUID) if USE_CUDA else features
#                 labels = labels.cuda(GPUID) if USE_CUDA else labels

#                 print('-------testing with testing data----------')
#                 results = lstm(features)
#                 predicts = []
#                 for i in range(features.size()[0]): #sequence length
#                     dist = results[i]
#                     topv, topi = dist.data.topk(1, 1)
#                     predicts.append(topi)
                
#                 error_count = 0 
#                 total = 0
#                 for i in range(features.size()[1]): #batch length 
#                     for j in range(features.size()[0]): #seq length
#                         if predicts[j][i,0] != labels.data[j,i]:
#                             error_count +=1
#                         total +=1
#                 print('error rate: {} / {} = {}'.format(error_count, total, error_count/total))
            
        if epoch % save_every ==0:
            # set model for evaluate
            encoder.eval()
            torch.save(encoder.state_dict(), open(os.path.join(save_path, 'epoch_{}_encoder.bin'.format(epoch)), 'wb'))
            decoder.eval()
            torch.save(decoder.state_dict(), open(os.path.join(save_path, 'epoch_{}_decoder.bin'.format(epoch)), 'wb'))
            print('--- save model ---')


def main():
    data_path = 'data'
    vids, vfeats, captions = read_training_data(data_path)
    assert len(vfeats) == len(captions)
    print('reading data finished')

    processed_captions = []
    for multi_caps in captions:
        new_caps = []
        for cap in multi_caps:
            tokens = cap.split()
            pcap = []
            for t in tokens:
                if len(t)>1:
                    t = t.strip('.!,()')
                pcap.append(t.lower())
            new_caps.append(pcap)
        processed_captions.append(new_caps)

    print(processed_captions[0])

    all_captions = []
    for multi_caps in processed_captions:
        all_captions.extend(multi_caps)
    seq_len = list(map(len, all_captions))
    print('max cap length {}'.format(max(seq_len)))

    idx2term, term2idx = get_dictionary(all_captions)
    print(idx2term[0:20])
    print(list(term2idx.items())[0:20])
    print(len(idx2term))

    tmp = []
    for multi_caps in processed_captions:
        idx_cap_lst =[]
        for cap in multi_caps:
            cap_idx = [START_IDX] + [term2idx[term] for term in cap] + [END_IDX]
            cap_idx = pad_seq(cap_idx, TARGET_MAX_LENGTH, 'l')
            idx_cap_lst.append(cap_idx)
        tmp.append(idx_cap_lst)
    processed_captions = tmp
    print(processed_captions[0])

#     train_fseq, test_fseq = {}, {}
#     train_lseq, test_lseq = {}, {}
#     shuffled_keys = list(fsequences.keys())
#     random.shuffle(shuffled_keys)
#     for i, skey in enumerate(shuffled_keys):
#         if i < TESTING_NUM:
#             test_fseq[skey] = fsequences[skey]
#             test_lseq[skey] = lsequences[skey]
#         else:
#             train_fseq[skey] = fsequences[skey]
#             train_lseq [skey] = lsequences[skey]
#     print("training num = {} {} testing num ={} {}".format(len(train_fseq), len(train_lseq), len(test_fseq), len(test_lseq)))
    
    input_dim = vfeats[0].shape[1]
    output_dim = len(idx2term)
    PARAMS['input_dim'] = input_dim
    PARAMS['output_dim'] = output_dim
    
    encoder = EncoderRNN(input_dim = input_dim, 
                                             hidden_dim = HIDDEN_DIM, 
                                             n_layers = N_LAYER, 
                                             bidirectional = BIDIRECTIONAL, 
                                             dropout_rate = DROPOUT_RATE)
                    
    decoder = AttnDecoderRNN(hidden_dim = HIDDEN_DIM, 
                                                    output_dim = output_dim, 
                                                     n_layers = N_LAYER, 
                                                     dropout = DROPOUT_RATE, 
                                                     max_length=INPUT_MAX_LENGTH)
        
    print(encoder)
    print(decoder)
                    
    if USE_CUDA:
        print('using CUDA models')
        encoder = encoder.cuda(GPUID)
        decoder = decoder.cuda(GPUID)

    tmpf = vfeats[0:5]
    tmpl = processed_captions[0:5]
    
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
    pickle.dump((idx2term), open(os.path.join(sub_path,'training_meta.pkl'), 'wb'))
    json.dump(PARAMS, open(os.path.join(sub_path, 'params.json'),'w'))

    # main training process
    trainEpochs(encoder = encoder,
                         decoder = decoder, 
                          vfeats = tmpf, 
                          processed_captions = tmpl, 
                          learning_rate = LEARNING_RATE, 
                          n_epochs = NUM_EPOCH, 
                          idx2term = idx2term, 
                          print_every = PRINT_EVERY, 
                          test_every = TEST_EVERY, 
                          save_every = SAVE_EVERY, 
                          save_path = sub_path, 
                          test_pairs=None)

    # set model for evaluation
    encoder.eval()
    decoder.eval

    torch.save(encoder.state_dict(), open(os.path.join(sub_path, 'encoder.bin'), 'wb'))
    torch.save(decoder.state_dict(), open(os.path.join(sub_path, 'decoder.bin'), 'wb'))
    
    print('All of models are trained and saved to {}'.format(sub_path))



class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim,  n_layers=1, bidirectional = False, dropout_rate=0):
        super(EncoderRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.direction = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout = dropout_rate)

    def forward(self, input_, hidden=None, cell=None):
        if hidden is None:
            hidden = self.initHidden(input_.size()[1])
        if cell is None:
            cell = self.initHidden(input_.size()[1])
        
        seq_len = input_.size()[0]
        n_sample = input_.size()[1]

        output = input_
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        
        return output, (hidden, cell)
    
    def initHidden(self, n_sample):
        result = Variable(torch.zeros(self.n_layers * self.direction, n_sample, self.hidden_dim))
        if USE_CUDA:
            return result.cuda(GPUID)
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers=1, dropout=0.1, max_length=INPUT_MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_dim, self.hidden_dim)
        self.attn = nn.Linear(self.hidden_dim * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=n_layers, dropout = dropout)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.LogSoftmax = nn.LogSoftmax()

    def forward(self, input_, hidden, cell, encoder_outputs):
        embedded = self.embedding(input_)[:, 0, :] #batch*embed_dim
        embedded = self.dropout(embedded)
        
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded,  hidden[0,:,:]), 1))) # batch * max_length
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), #batch * 1 * max_length(seq)
                                 torch.transpose(encoder_outputs, 0, 1)) # batch * max_length(seq) * hidden_dim 
        attn_applied = attn_applied[:, 0, :] #batch*hidden_dim

        output = torch.cat((embedded, attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = self.LogSoftmax(self.out(output[-1, :, :]))

        return output, hidden, cell, attn_weights

    def initHidden(self, n_sample):
        result = Variable(torch.zeros(1, n_sample, self.hidden_dim))
        if USE_CUDA:
            return result.cuda(GPUID)
        else:
            return result

if __name__ == '__main__':
    main()
