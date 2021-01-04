# -*- coding: utf-8 -*-
# Evaluation Model
import re
import os
import pickle
import logging
import numpy as np
import pandas as pd
from numpy.linalg import norm
#import matplotlib.pylab as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

import torch

#############################################################
################# Subword Information
#############################################################
def custom_cosine_similarity(x, y)-> int: 
    """ 1D tensor input """
    return x.dot(y) / (norm(x)* norm(y) + 1e-15) # l2 norm in paper
"""
Evaluation datasets that are not part of training set;
Represent them using null vectors (sisg-).
Compute vectors for unseen words by summing the n-gram vectors (sisg).
"""
def cosine_sim_rg_sub(rg65, model, word2idx):
    rm_cnt=0
    sim_human = []
    sim_lst = []
    for i in range(len(rg65)):
        try:
            w1 = '<' + rg65[i].split(';')[0] + '>'
            w2 = '<' + rg65[i].split(';')[1] + '>'
            x = model.embedding_i.weight[word2idx[w1]]
            y = model.embedding_i.weight[word2idx[w2]]
            sim = custom_cosine_similarity(x.cpu().detach().numpy(), y.cpu().detach().numpy())
            sim_lst.append(sim)
            sim_human.append(float(rg65[i].split(';')[2].replace(' ','').replace('\n','')))
        except Exception as e:
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human

def cosine_sim353_sub(sim353, model, word2idx):
    rm_cnt=0
    sim_human = []
    sim_lst = []

    for lst in sim353[11:]:
        line = lst.split('\t')
        line[1] = '<' + line[1] + '>'
        line[2] = '<' + line[2] + '>'
        try:
            x = model.embedding_i.weight[word2idx[line[1]]]
            y = model.embedding_i.weight[word2idx[line[2]]]
            sim = custom_cosine_similarity(x.cpu().detach().numpy(), y.cpu().detach().numpy())
            sim_lst.append(sim)
            sim_human.append(float(line[3].replace("\n","")))
        except Exception as e:
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human

def cosine_sim_simlex_sub(simlex, model, word2idx):
    rm_cnt=0
    sim_lst=[]
    sim_human=[]
    for line in simlex:
        line = line.split('\t')
        line[0] = '<' + line[0] + '>'
        line[1] = '<' + line[1] + '>'
        try:
            x = model.embedding_i.weight[word2idx[line[0]]]
            y = model.embedding_i.weight[word2idx[line[1]]]
            sim = custom_cosine_similarity(x.cpu().detach().numpy(), y.cpu().detach().numpy())
            sim_lst.append(sim)
            sim_human.append(line[3])
        except Exception as e:
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human

#############################################################
################# Word2Vec
#############################################################

def cosine_sim_simrel(sim_rel, model, word2idx):
    rm_cnt=0
    sim_human = []
    sim_lst = []

    for lst in sim_rel:
        line = lst.split('\t') 
        try:
            x = model.embedding_i.weight[word2idx[line[0]]]
            y = model.embedding_i.weight[word2idx[line[1]]]
            sim = custom_cosine_similarity(x.cpu().detach().numpy(), y.cpu().detach().numpy())
            sim_lst.append(sim)
            sim_human.append(float(line[2].replace("\n","")))
        except Exception as e:
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human

def cosine_sim_simlex(simlex, model, word2idx):
    rm_cnt=0
    sim_lst=[]
    sim_human=[]
    for line in simlex:
        line = line.split('\t')
        try:
            x = model.embedding_i.weight[word2idx[line[0]]]
            y = model.embedding_i.weight[word2idx[line[1]]]
            sim = custom_cosine_similarity(x.cpu().detach().numpy(), y.cpu().detach().numpy())
            sim_lst.append(sim)
            sim_human.append(line[3])
        except Exception as e:
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human

def cosine_sim_men_(men, model, word2idx):
    rm_cnt=0
    sim_lst=[]
    sim_human=[]
    for word1, word2, num in men:
        w1 = word1.replace('-n','').replace('-j','').replace('-v','')
        w2 = word2.replace('-n','').replace('-j','').replace('-v','')
        try:
            x = model.embedding_syn0.weight[word2idx[w1]]
            y = model.embedding_syn0.weight[word2idx[w2]]
            sim = custom_cosine_similarity(x.cpu().detach().numpy(), y.cpu().detach().numpy())
            sim_lst.append(sim)
            sim_human.append(float(num.replace("\n","")))
        except Exception as e:
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human


def cosine_sim_men(men, model, word2idx):
    rm_cnt=0
    sim_lst=[]
    sim_human=[]
    for word1, word2, num in men:
        w1 = word1.replace('-n','').replace('-j','').replace('-v','')
        w2 = word2.replace('-n','').replace('-j','').replace('-v','')
        try:
            x = model.embedding_i.weight[word2idx[w1]]
            y = model.embedding_i.weight[word2idx[w2]]
            sim = custom_cosine_similarity(x.cpu().detach().numpy(), y.cpu().detach().numpy())
            sim_lst.append(sim)
            sim_human.append(float(num.replace("\n","")))
        except Exception:
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human


def cosine_sim_rg_(rg65, model, word2idx):
    rm_cnt=0
    sim_human = []
    sim_lst = []
    for i in range(len(rg65)):
        try:
            x = model.embedding_syn0.weight[word2idx[rg65[i].split(';')[0]]]
            y = model.embedding_syn0.weight[word2idx[rg65[i].split(';')[1]]]
            sim = custom_cosine_similarity(x.cpu().detach().numpy(), y.cpu().detach().numpy())
            sim_lst.append(sim)
            sim_human.append(float(rg65[i].split(';')[2].replace(' ','').replace('\n','')))
        except Exception:
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human


def cosine_sim_rg(rg65, model, word2idx):
    rm_cnt=0
    sim_human = []
    sim_lst = []
    for i in range(len(rg65)):
        try:
            x = model.embedding_i.weight[word2idx[rg65[i].split(';')[0]]]
            y = model.embedding_i.weight[word2idx[rg65[i].split(';')[1]]]
            sim = custom_cosine_similarity(x.cpu().detach().numpy(), y.cpu().detach().numpy())
            sim_lst.append(sim)
            sim_human.append(float(rg65[i].split(';')[2].replace(' ','').replace('\n','')))
        except Exception:
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human

def cosine_sim353(sim353, model, word2idx):
    rm_cnt=0
    sim_human = []
    sim_lst = []

    for lst in sim353[11:]:
        line = lst.split('\t')
        try:
            x = model.embedding_i.weight[word2idx[line[1]]]
            y = model.embedding_i.weight[word2idx[line[2]]]
            sim = custom_cosine_similarity(x.cpu().detach().numpy(), y.cpu().detach().numpy())
            sim_lst.append(sim)
            sim_human.append(float(line[3].replace("\n","")))
        except:
            rm_cnt+=1
    print("Removed eval dataset count: ", rm_cnt)
    return sim_lst, sim_human

if __name__ == '__main__':
    
    SEED = 42 # predictable random variables
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
    logger =logging.getLogger(__name__)

    # path_rg65 = '/home/ubuntu/nlp/tutorial/dataset/wrd/eval/RG65.csv'
    # path_sim353 = '/home/ubuntu/nlp/tutorial/dataset/wrd/eval/wordsim353_sim_rel/wordsim353_annotator1.txt'
    
    model_path = 'result_0810'
    name = 'skip_gram'
    hidden = 100
    lr = 0.001
    key = '16'

    with open(f'./{model_path}/vocab.pkl', 'rb') as reader:
        vocab = pickle.load(reader)
        word2idx, idx2word = vocab
    
    vocab_size = len(vocab[0]) # 16.7M

    model = NCEloss(vocab_size, hidden)
    model.load_state_dict(torch.load(f'./{model_path}/{name}_{hidden}_{lr}_{key}.pt'))
    model.eval()
    logger.info("Model loaded")


