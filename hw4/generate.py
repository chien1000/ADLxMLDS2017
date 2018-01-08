
# coding: utf-8

# In[34]:



import random
import time
import pickle
import json
import os
import scipy.misc as misc


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


# In[20]:


from train import Generator, Discriminator, tags_to_ids, to_one_hot, to_variable


# In[29]:


USE_CUDA = torch.cuda.is_available()


# In[28]:


seed = 1225
random.seed(seed)
torch.manual_seed(seed)
if USE_CUDA:
    torch.cuda.manual_seed_all(seed)


# In[36]:


def draw_outputs(imgs, names, img_path):
    for img_id in range(imgs.size()[0]):
        to_draw = imgs[img_id,:,:,:].permute(1,2,0).cpu().data.numpy()
        save_path = os.path.join(img_path, names[img_id])
        misc.imsave(save_path, to_draw)


# In[32]:


def generate(generator, noise_dim, text_var):
    noise = Variable(torch.randn(text_var.size(0), noise_dim))
    if USE_CUDA:
        noise = noise.cuda()
    noise = noise.view(noise.size(0), noise_dim, 1, 1)
    fake_img = generator(text_var, noise)
    
    return fake_img


# In[46]:


def main(model_dir):
    trainging_meta_path = os.path.join(model_dir,'training_meta.pkl')
    # D_path = os.path.join(model_dir, 'discriminator.bin')
    G_path = os.path.join(model_dir, 'generator.bin')
    params_path = os.path.join(model_dir, 'params.json')

    tmeta = pickle.load(open(trainging_meta_path,'rb'))
    hair_colors, eye_colors = tmeta[0], tmeta[1]
    hair_colors_dict = { c:i for i, c in enumerate(hair_colors)}
    eye_colors_dict = { c:i for i, c in enumerate(eye_colors)}

    params = json.load(open(params_path,'r'))

    embed_dim = len(hair_colors) + len(eye_colors)
    project_dim = params['PROJECT_DIM']
    noise_dim = params['NOISE_DIM']

    # discriminator = Discriminator(embed_dim, project_dim)
    generator = Generator(embed_dim, project_dim)

    # discriminator.load_state_dict(torch.load(D_path, map_location={'cuda:0': 'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu'}))
    generator.load_state_dict(torch.load(G_path, map_location={'cuda:0': 'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu'}))
    if USE_CUDA:
        generator = generator.cuda()


    img_path = 'samples/'
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    test_tags = []
    with open('data/sample_testing_text.txt') as fin:
        for line in fin.readlines():
            tags = line.strip().split(',')[1]
            test_tags.append(tags)

    one_hots = [to_one_hot(tags_to_ids(tags)) for tags in test_tags]
    text_var = to_variable(one_hots)

    for i in range(5):
        n = text_var.size()[0]
        names = ['sample_{}_{}.jpg'.format(j+1, i+1) for j in range(n)]

        imgs = generate(generator, noise_dim, text_var)
        draw_outputs(imgs, names, img_path)


# In[47]:


# model_dir = 'models/_NOISE_100_LR_0.0002_MOMEN_0.5_L1_50_L2_100'
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
        main(model_dir)

