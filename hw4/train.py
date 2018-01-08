
# coding: utf-8


import skimage
import skimage.io
import skimage.transform
import scipy
import scipy.misc as misc
import os


import random
import time
import pickle
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


NUM_EPOCHS = 1000
NOISE_DIM = 100
PROJECT_DIM = 64
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
MOMENTUM = 0.5
L1_COEF = 0
L2_COEF = 0
PRINT_EVERY = 50
DRAW_EVERY = 300
SAVE_EVERY = 5
SAVE_PATH ='models/'
SAVE_PREFIX = ''
PARAMS = {'NUM_EPOCHS':NUM_EPOCHS, 'NOISE_DIM':NOISE_DIM,'PROJECT_DIM':PROJECT_DIM,'LEARNING_RATE':LEARNING_RATE,'MOMENTUM':MOMENTUM,  
        'L1_COEF':L1_COEF, 'L2_COEF':L2_COEF}
USE_CUDA = torch.cuda.is_available()


hair_colors = ['orange', 'white', 'aqua','gray', 'green', 'red', 'purple', 'pink', 
             'blue', 'black', 'brown', 'blonde' ]
hair_colors_dict = { c:i for i, c in enumerate(hair_colors)}

eye_colors = ['gray', 'black', 'orange', 'pink', 'yellow', 'aqua', 'purple',
              'green', 'brown', 'red', 'blue']
eye_colors_dict = { c:i for i, c in enumerate(eye_colors)}

# print(hair_colors)
# print(hair_colors_dict)
# print(eye_colors)
# print(eye_colors_dict)


def tags_to_ids(tags):
    h_id = -1
    e_id = -1
    elements = tags.split()
    for i in range(0,len(elements), 2):
        part = elements[i+1]
        color = elements[i]

        if  part == 'hair':
            h_id = hair_colors_dict[color]
        
        elif part == 'eyes':
            e_id = eye_colors_dict[color]
    
    return [e_id, h_id]


def to_one_hot(ids):
    one_hot = np.zeros(len(hair_colors)+len(eye_colors))
    if ids[0] != -1:
        one_hot[ids[0]]=1
    if ids[1]!= -1:
        one_hot[len(eye_colors)+ids[1]]=1
    return one_hot

  
# batch_ids = [[1,2],[-1,3],[2,-1]]
# eyes_ids = torch.Tensor([ids[0] for ids in batch_ids])
# hair_ids = torch.Tensor([ids[1] for ids in batch_ids])

# eyes_mask = torch.ByteTensor(tuple(map(lambda i: int(i != -1), eyes_ids)))
# hairs_mask = torch.ByteTensor(tuple(map(lambda i: int(i != -1), hair_ids)))
# print(eyes_ids)
# print(eyes_mask)
# print(eyes_ids[eyes_mask])
# one_hot = torch.zeros((3,len(eye_colors)))
# print(one_hot)
# print(one_hot[eyes_mask,])
# one_hot[eyes_mask].scatter_(dim=1, index=eyes_ids[eyes_mask], src=1.)

# """
# ids: (list, ndarray) shape:[batch_size]
# out_tensor:FloatTensor shape:[batch_size, depth]
# """
# if not isinstance(ids, (list, np.ndarray)):
#     raise ValueError("ids must be 1-D list or array")
# ids = torch.LongTensor(ids).view(-1,1)
# out_tensor.zero_()
# out_tensor.scatter_(dim=1, index=ids, src=1.)
# out_tensor.scatter_(1, ids, 1.0)

def one_hot_to_text(one_hot):

    e_color = '_'
    h_color = '_'

    pos = np.argwhere(one_hot==1)

    if pos.shape[1] ==1:
        if pos[0,0] < len(eye_colors):
            e_color = eye_colors[pos[0,0]]
        else:
            h_id = pos[0,0] - len(eye_colors)
            h_color = hair_colors[h_id]
            
    elif pos.shape[1] ==2:
        e_color = eye_colors[pos[0,0]]
        h_id = pos[0,1] - len(eye_colors)
        h_color = hair_colors[h_id]

    
    return '{}-{}'.format(e_color, h_color)


class Generator(nn.Module):
    def __init__(self, embed_dim, project_embed_dim):
        super(Generator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = NOISE_DIM
        self.embed_dim = embed_dim
        self.projected_embed_dim = project_embed_dim
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = 64

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        # based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.netG = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
             # state size. (num_channels) x 64 x 64
            )


    def forward(self, embed_vector, z):

        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([projected_embed, z], 1)
        output = self.netG(latent_vector)

        return output

class Discriminator(nn.Module):
    def __init__(self, embed_dim, projected_embed):
        super(Discriminator, self).__init__()
#         self.image_size = 64
        self.num_channels = 3
        self.embed_dim = embed_dim
        self.projected_embed_dim = projected_embed
        self.ndf = 64
        
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.projected_embed_dim),
            nn.BatchNorm1d(num_features = self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        self.netD_1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

#         self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

        self.netD_2 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8 + self.projected_embed_dim, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )   

    def forward(self, img, embed):
        img_intermediate = self.netD_1(img)
        
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2,  3, 0, 1)
        embed_concat = torch.cat([img_intermediate, replicated_embed], 1)
        embed_concat = self.netD_2(embed_concat)

        return embed_concat.view(-1, 1).squeeze(1) , img_intermediate

def to_variable(values_list):
    if isinstance(values_list, torch.Tensor):
        var = Variable(values_list)
    if isinstance(values_list, list):
        var = np.array(values_list)
        var = Variable(torch.from_numpy(var).float())
    
    return var


def to_batch_variable(batch):
   
    texts = to_variable([to_one_hot(pair[0]) for pair in batch])

    imgs = to_variable([pair[1] for pair in batch])
    imgs = imgs.permute(0,3,1,2)
    
    return texts, imgs



def make_training_data(text_img_pairs):
    n_data = len(text_img_pairs)
    steps = list(range(0, n_data+1, int(BATCH_SIZE/2)))
    if steps[-1] < n_data:
        steps.append(n_data)    
    # print(steps)

    correct_batches = []
    wrong_batches = []

    for i in range(len(steps)-1):
        st = steps[i]
        ed = steps[i+1]
        batch_pairs= text_img_pairs[st:ed]
        wrong_pairs = []

        text_pool= set([tuple(p[0]) for p in batch_pairs])
    #     print(text_pool)

        for text, img in batch_pairs:
            tmp = text_pool.copy()
            tmp.remove(tuple(text))

            wrong_text = random.choice(tuple(tmp))
            wrong_pairs.append((wrong_text, img))
        
        
        correct_batches.append(to_batch_variable(batch_pairs))
        wrong_batches.append(to_batch_variable(wrong_pairs))
    
    return correct_batches, wrong_batches
    


def train_gan(correct_batches, wrong_batches, save_path, save_path_img):

    embed_dim = len(hair_colors) + len(eye_colors)
    discriminator = Discriminator(embed_dim, PROJECT_DIM)
    generator = Generator(embed_dim, PROJECT_DIM)
    if USE_CUDA:
        discriminator = discriminator.cuda()
        generator = generator.cuda()
    criterion = nn.BCELoss()
    l2_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    optimD = torch.optim.Adam(discriminator.parameters(), lr = LEARNING_RATE, betas=(MOMENTUM, 0.999))
    optimG = torch.optim.Adam(generator.parameters(), lr = LEARNING_RATE, betas=(MOMENTUM, 0.999))

    iteration = 0
    
    cw_batches = list(zip(correct_batches, wrong_batches))
    for epoch in range(NUM_EPOCHS):
        batch = 0
        random.shuffle(cw_batches)
        
        for cbatch, wbatch in cw_batches:
            batch+=1
            print('batch {}/{}'.format(batch, len(correct_batches)))
            iteration += 1

            ctext, cimg = cbatch[0], cbatch[1]
            wtext, wimg = wbatch[0], wbatch[1]

            if USE_CUDA:
                ctext, cimg, wtext, wimg = ctext.cuda(), cimg.cuda(), wtext.cuda(), wimg.cuda()

            real_labels = torch.ones(cimg.size(0))
            fake_labels = torch.zeros(wimg.size(0))

            # ======== One sided label smoothing ==========
            # Helps preventing the discriminator from overpowering the
            # generator adding penalty when the discriminator is too confident
            # =============================================
            smoothed_real_labels = torch.FloatTensor(real_labels.numpy() - 0.1)

            real_labels = Variable(real_labels)
            smoothed_real_labels = Variable(smoothed_real_labels)
            fake_labels = Variable(fake_labels)

            if USE_CUDA:
                real_labels = real_labels.cuda()
                smoothed_real_labels = smoothed_real_labels.cuda()
                fake_labels = fake_labels.cuda()

            # Train the discriminator
            discriminator.zero_grad()

            ##right image, right text
            real_score, _ = discriminator(cimg, ctext)
            real_loss = criterion(real_score, smoothed_real_labels)

            ##right image, wrong text
            wrong_score, _ = discriminator(wimg, wtext)
            wrong_loss = criterion(wrong_score, fake_labels)

            ##fake image
            noise = Variable(torch.randn(cimg.size(0), NOISE_DIM))
            if USE_CUDA:
                noise = noise.cuda()
            noise = noise.view(noise.size(0), NOISE_DIM, 1, 1)
            fake_img = generator(ctext, noise)
            fake_score, _ = discriminator(fake_img, ctext)
            fake_loss = criterion(fake_score, fake_labels)

            total_d_loss = real_loss + fake_loss + wrong_loss

            total_d_loss.backward()
            optimD.step()

            # Train the generator
            generator.zero_grad()
            noise = Variable(torch.randn(cimg.size(0), NOISE_DIM))
            if USE_CUDA:
                noise = noise.cuda()
            noise = noise.view(noise.size(0), NOISE_DIM, 1, 1)
            fake_img = generator(ctext, noise)
            fake_score, activation_fake = discriminator(fake_img, ctext)
            _, activation_real = discriminator(cimg, ctext)

            activation_fake = torch.mean(activation_fake, 0)
            activation_real = torch.mean(activation_real, 0)


            #======= Generator Loss function============
            # This is a customized loss function, the first term is the regular cross entropy loss
            # The second term is feature matching loss, this measure the distance between the real and generated
            # images statistics by comparing intermediate layers activations
            # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
            # because it links the embedding feature vector directly to certain pixel values.
            #===========================================
            g_loss = criterion(fake_score, real_labels)
            + L2_COEF * l2_loss(activation_fake, activation_real.detach()) 
            + L1_COEF * l1_loss(fake_img, cimg)

            g_loss.backward()
            optimG.step()

            if iteration % PRINT_EVERY== 0:
                print('Epoch {}: d_loss {}, g_loss {}, real_score {}, fake_score {}'.format(epoch, 
                                    total_d_loss.data[0], g_loss.data[0], 
                                    real_score.mean().data[0], fake_score.mean().data[0]))

            if iteration % DRAW_EVERY ==0:
                random_draw = np.random.choice(fake_img.size()[0], 5, replace=False)
                for i, img_id in enumerate(random_draw):
    #                 print(fake_img[img_id,:,:,:].permute(1,2,0).size())
                    to_draw = fake_img[img_id,:,:,:].permute(1,2,0).cpu().data.numpy()
                    text_one_hot = ctext[img_id,].cpu().data.numpy()
                    text_colors = one_hot_to_text(text_one_hot)
                    img_path = os.path.join(save_path_img, '{}-{}-{}.jpg'.format(iteration, i, text_colors))
                    misc.imsave(img_path, to_draw)

        if epoch % SAVE_EVERY == 0:
            torch.save(discriminator.state_dict(), open(os.path.join(save_path, 'epoch_{}_discriminator.bin'.format(epoch)), 'wb'))
            torch.save(generator.state_dict(), open(os.path.join(save_path, 'epoch_{}_generator.bin'.format(epoch)), 'wb'))


def main():
    ts = "%d"%(time.time())
    params = '{}_NOISE_{}_LR_{}_MOMEN_{}_L1_{}_L2_{}'.format(SAVE_PREFIX, NOISE_DIM, LEARNING_RATE, MOMENTUM, L1_COEF, L2_COEF)
    sub_path = os.path.join(SAVE_PATH,params)
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
        sub_path_img = os.path.join(sub_path, 'img')
        os.makedirs(sub_path_img)
    else:
        sub_path = sub_path+ '_' + ts
        os.makedirs(sub_path)
        sub_path_img = os.path.join(sub_path, 'img')
        os.makedirs(sub_path_img)

    pickle.dump((hair_colors, eye_colors), open(os.path.join(sub_path,'training_meta.pkl'), 'wb'))
    json.dump(PARAMS, open(os.path.join(sub_path, 'params.json'),'w'))
    
    #load text
    texts = []
    with open('data/tags_clean.csv','r') as fin:
        for line in fin.readlines():
            content = line.split(',')[1]
            tags = content.split('\t')
            target_tags = ''
            for tag in tags:
                if 'hair' in tag or 'eyes' in tag:
                    tag = tag.split(':')[0]
                    elements = tag.split(' ')
                    if elements[0] in hair_colors_dict or elements[0] in eye_colors_dict:
                        target_tags = target_tags + ' ' + tag
            texts.append(target_tags)
    print(texts[:100])

    print(len(texts))
    a = list(filter(lambda x: len(x)>0, texts))
    print(len(a))
    
    #load images 
    text_img_pairs = []
    for i, tag in enumerate(texts):
        if len(tag)  == 0:
            continue

        img_path = os.path.join('data/faces/', '{}.jpg'.format(i))
        img = skimage.io.imread(img_path)
        img = skimage.transform.resize(img, (64,64))

        ids = tags_to_ids(tag)
        text_img_pairs.append((ids, img))


    correct_batches, wrong_batches = make_training_data(text_img_pairs)
    print('----start training----')
    train_gan(correct_batches, wrong_batches, sub_path, sub_path_img)

if __name__ == '__main__':
    main()