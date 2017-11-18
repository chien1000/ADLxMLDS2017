# coding: utf-8


import torch

import pickle
from os import listdir
from os.path import isfile, join, basename
import os
import json
from collections import defaultdict
import time

import numpy as np
from train import AttnDecoderRNN, DecoderRNN, EncoderRNN, evaluate, get_variable_from_seq
# from models import LSTMRecognizer, CNN_LSTMRecognizer

def read_features(data_path):
    feat_path = join(data_path,'testing_data/feat')
    feat_files = [join(feat_path, f) for f in listdir(feat_path) if isfile(join(feat_path, f))]
    
    vfeats = []
    vids = []
    for feat_f in feat_files:
        vid = os.path.splitext(basename(feat_f))[0]
        feat = np.load(feat_f)

        vfeats.append(feat)
        vids.append(vid)
    
    return vids, vfeats

def read_features_specified(data_path, specified_list):
    feat_path = join(data_path,'testing_data/feat')
    feat_files = [join(feat_path, f+'.npy') for f in specified_list]
    
    vfeats = []
    vids = []
    for feat_f in feat_files:
        vid = os.path.splitext(basename(feat_f))[0]
        feat = np.load(feat_f)

        vfeats.append(feat)
        vids.append(vid)
    
    return vids, vfeats


def make_prediction(model, fsequences, idx_to_label, pad_idx, map48_to39_path, map_phone_to_char_path):
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


        try:
            predicts.remove(pad_idx)
        except Exception as e:
            pass
            
        labels = [idx_to_label[pidx] for pidx in predicts]
#         print(labels)
        # try:
        #     labels.remove('SOS')
        #     labels.remove('EOS')
        #     labels.remove('PAD')

        # except Exception as e:
        #     pass
        
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

    encoder_path = os.path.join(model_dir, 'encoder.bin')
    decoder_path = os.path.join(model_dir, 'decoder.bin')
    params_path = os.path.join(model_dir, 'params.json')

    tmeta = pickle.load(open(trainging_meta_path,'rb'))
    # print(tmeta)
    idx2term = tmeta #[0]
    # print(idx2term)
    
    params = json.load(open(params_path,'r'))

    encoder = EncoderRNN(input_dim = params['input_dim'], 
                                             hidden_dim = params['HIDDEN_DIM'], 
                                             n_layers = params['N_LAYER'], 
                                             bidirectional = params['BIDIRECTIONAL'], 
                                             dropout_rate = params['DROPOUT_RATE'])
    if params['USE_ATTENTION']:              
        decoder = AttnDecoderRNN(hidden_dim = params['HIDDEN_DIM'], 
                                                        output_dim = params['output_dim'], 
                                                         n_layers = params['N_LAYER'], 
                                                         dropout = params['DROPOUT_RATE'], 
                                                         max_length = params['INPUT_MAX_LENGTH']) 
    else:
        decoder = DecoderRNN(hidden_dim = params['HIDDEN_DIM'], 
                                                        output_dim = params['output_dim'], 
                                                         n_layers = params['N_LAYER'], 
                                                         dropout = params['DROPOUT_RATE'])
        


    encoder.load_state_dict(torch.load(encoder_path, map_location={'cuda:0': 'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu'}))
    decoder.load_state_dict(torch.load(decoder_path, map_location={'cuda:0': 'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu'}))
    

    encoder.eval()
    decoder.eval()
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    vids, vfeats = read_features(data_dir)
    # specified_list = ['klteYv1Uv9A_27_33.avi', '5YJaS2Eswg0_22_26.avi', 'UbmZAe5u5FI_132_141.avi', 'JntMAcTlOF0_50_70.avi', 'tJHUH9tpqPg_113_118.avi']
    # vids, vfeats = read_features_specified(data_dir, specified_list)
    # print(vids)

    predict_lines = []
    for vid, vfeat in zip(vids, vfeats):
        eval_var = get_variable_from_seq(vfeat, 'f').unsqueeze(1)

        pred_words = evaluate(encoder, decoder, eval_var, idx2term, params['TARGET_MAX_LENGTH'])
        
        if 'PAD' in pred_words:
            # print(pred_words[-1]=='PAD')
            # print(len(pred_words[-1]))
            pred_words.remove('PAD')
        if 'START' in pred_words:
            pred_words.remove('START')
        if 'END' in pred_words:
            pred_words.remove('END')

        predict_lines.append('{},{}'.format(vid, ' '.join(pred_words).replace('PAD','').capitalize()))
    
    with open( output_fname, 'w') as fout:
        # fout.write('id,phone_sequence\n')
        fout.write('\n'.join(predict_lines))
    

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        model_dir = sys.argv[1] 
        data_dir = sys.argv[2]
        output_fname = sys.argv[3]
        main(model_dir, data_dir, output_fname)

    else:
        print('incorrect argument number')
