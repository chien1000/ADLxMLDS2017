# coding: utf-8


import torch

import pickle
import os
import json
from collections import defaultdict
import time

import numpy as np
from rnn import read_ark, get_variable_from_seq
from models import LSTMRecognizer, CNN_LSTMRecognizer


def make_sequences(data, features_mean, features_std):
#     all_sent = set(k[0] for k in data.keys())
    sort_keys = sorted(data.keys(), key = lambda x: (x[0], x[1], int(x[2])))
#     print(sort_keys[:100])

    fsequences = defaultdict(list)

    for iid in sort_keys:
        speaker_id = iid[0]
        sent_id = iid[1]

        info = data[iid]
        features = np.array(list(map(float, info['features'])))
        features = (features - features_mean)/features_std

        # if info['gender'] == 'f':
        #     features.append(0)
        # else:
        #     features.append(1)
    
        fsequences[(speaker_id, sent_id)].append(features)
                
    return fsequences
    
# def make_input_tensors(fsequences):
# #     input_tensor = [get_variable_from_seq(seq,'f').unsqueeze(1) for seq in fsequences]
# #     input_tensor = torch.cat(input_tensor,1)
#     batches = defaultdict(list)
#     input_tensors = []

#     for sent_uid in fsequences:
#         fseq = fsequences[sent_uid]
#         fseq_var = get_variable_from_seq(fseq, 'f')
#         seq_len = len(fseq) 
#         batches[seq_len].append(fseq_var)

#     for batch in batches.values():
# #         print(len(batch))
#         fbatch = [seq.unsqueeze(1) for seq in batch]
#         fbatch = torch.cat(fbatch, 1)
#         input_tensors.append(fbatch)
    
#     return input_tensors

def trim_phone(phone_seq, silent_token = 'sil'):
    new_phone = [phone_seq[0]]
    p_frame = phone_seq[0]
    for frame in phone_seq[1:]:
        if frame != p_frame:
            new_phone.append(frame)
        p_frame = frame
    
    idx = 0
    while new_phone[idx] == silent_token:
        idx+=1
    
    idx2 = -1
    while new_phone[idx2] == silent_token:
        idx2 -=1
        
    new_phone = new_phone[idx:idx2+1]
    
    return new_phone

def map48_to39(map_path, phone_seq):
#     map_path = 'data/phones/48_39.map'
    mapping = {}
    with open(map_path, 'r') as fin:
        for line in fin.readlines():
            ele = line.strip().split('\t')
            mapping[ele[0]] = ele[1]
    
    new_seq = [mapping[p] for p in phone_seq]
    return new_seq

def map_phone_to_char(map_path, phone_seq):
    mapping = {}
    with open(map_path, 'r') as fin:
        for line in fin.readlines():
            ele = line.strip().split('\t')
            mapping[ele[0]] = ele[2]
    
    new_seq = [mapping[p] for p in phone_seq]
    return new_seq

def make_prediction(model, fsequences, idx_to_label, map48_to39_path, map_phone_to_char_path):
    predict_lines = []
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    model.eval()
    for usid, seq in fsequences.items():
        seq_var = get_variable_from_seq(seq, 'f')
        seq_var = seq_var.unsqueeze(1)
        if use_cuda:
            seq_var = seq_var.cuda()
#         print(seq_var.size())
        results = model(seq_var)
        predicts = []
        for i in range(seq_var.size()[0]): #sequence length
            dist = results[i]
            topv, topi = dist.data.topk(1, 1)
            predicts.append(topi[0,0])
#         print(predicts)

        labels = [idx_to_label[pidx] for pidx in predicts]
#         print(labels)
        try:
            labels.remove('SOS')
            labels.remove('EOS')
            labels.remove('PAD')

        except Exception as e:
            pass
        
        labels = map48_to39(map48_to39_path, labels)
#         print(labels)

        labels = trim_phone(labels)
#         print(labels)

        labels = map_phone_to_char(map_phone_to_char_path, labels)
#         print(labels)

        phone_str = ''.join(labels)
        final = '{}_{},{}'.format(usid[0], usid[1], phone_str)
        predict_lines.append(final)

    return predict_lines


def write_outputs(lines):
    cur_time = str(time.time()).split('.')[0]
    with open('outputs/output_'+cur_time+'.txt', 'w') as fout:
        fout.write('id,phone_sequence\n')
        fout.write('\n'.join(lines))
    

def main(model_dir, data_dir, output_fname):

    trainging_meta_path = os.path.join(model_dir,'training_meta.pkl')
    model_path = os.path.join(model_dir, 'model.bin')
    params_path = os.path.join(model_dir, 'params.json')

    tmeta = pickle.load(open(trainging_meta_path,'rb'))
    idx_to_label, features_mean, features_std = tmeta[0], tmeta[1], tmeta[2]
    
    
    params = json.load(open(params_path,'r'))
    if params['NETWORK'] == 'RNN':
        model = LSTMRecognizer(input_dim = params['input_dim'], 
                                  hidden_dim = params['HIDDEN_DIM'], 
                                  output_dim = params['output_dim'], 
                                  n_layers = params['N_LAYER'],
                                  bidirectional = params['BIDIRECTIONAL'],
                                  dropout_rate = params['DROPOUT_RATE'])
    elif params['NETWORK'] == 'CRNN':
        model = CNN_LSTMRecognizer(input_dim = params['input_dim'],
                                    output_dim = params['output_dim'],
                                    hidden_dim = params['HIDDEN_DIM'],
                                    lstm_n_layers= params['N_LAYER'], 
                                    bidirectional = params['BIDIRECTIONAL'], 
                                    dropout_rate = params['DROPOUT_RATE'], 
                                    out_channels = params['CHANNELS'], 
                                    kernel_size = params['KERNEL_SIZE'], 
                                    stride = params['STRIDE'], 
                                    pooling_size = params['POOLING_SIZE'], 
                                    padding = params['CNN_PADDING'], 
                                    dilation = params['DILATION'])

    model.load_state_dict(torch.load(model_path, map_location={'cuda:0': 'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu'}))

    if params['FEATURE'] == 'fbank':
        test = read_ark(os.path.join(data_dir, 'fbank/test.ark'))
    elif params['FEATURE'] == 'mfcc':
        test = read_ark(os.path.join(data_dir, 'mfcc/test.ark'))

    for k in test:
        print(k)
        print(test[k])
        break

    fsequences = make_sequences(test,features_mean, features_std)

    map48_to39_path = os.path.join(data_dir, 'phones/48_39.map')
    map_phone_to_char_path = os.path.join(data_dir, '48phone_char.map')

    predict_lines = make_prediction(model, fsequences, idx_to_label, map48_to39_path, map_phone_to_char_path)
    
    with open( output_fname, 'w') as fout:
        fout.write('id,phone_sequence\n')
        fout.write('\n'.join(predict_lines))
    

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        model_dir = sys.argv[1] # 'models/_HS_128_EP_100_LR_0.01'
        data_dir = sys.argv[2]
        output_fname = sys.argv[3]
        main(model_dir, data_dir, output_fname)

    else:
        print('incorrect argument number')


#         做PADDING!!!!!!!
# 合併features...
