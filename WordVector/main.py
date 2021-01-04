import os
import re
import time
import json
import math
import pickle
import logging
import numpy as np
from tqdm.auto import tqdm
from collections import Counter
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam 
from scipy.stats import spearmanr

import config
from src.optim import get_cosine_schedule_with_warmup
from src.model import Word2VecDataset, my_collate, NCEloss, SoftmaxLoss
from src.eval import cosine_sim_rg, cosine_sim353, cosine_sim_men, cosine_sim_simrel, cosine_sim_simlex


###########################
##  Loading Config File  ##
###########################
SAVE_DIR = config.SAVE_PATH
SEED = config.SEED # predictable random variables
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

with open('hyperparams.json') as f:
    params = json.load(f, object_hook = lambda d: SimpleNamespace(**d))

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
logger =logging.getLogger(__name__)


if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

######################################################
############# Features Settings  #####################
######################################################

# Hyper-params
DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else 'cpu' # same as torch.device(0)

# model
KEY = time.asctime().split()[3][:2] + time.asctime().split()[3][3:5]+time.asctime().split()[3][6:8]

with open(config.SAVE_PATH + '/vocab01.pkl', 'rb') as r:
    word2idx, idx2word = pickle.load(r)
logger.info("vocab loaded")

with open(config.SAVE_PATH + '/unigram01.pkl', 'rb') as r:
    unigram, denominator = pickle.load(r)

######################################################
############## Evaluation Settings ###################
######################################################

path_rg65 = config.EVAL_ROOT_PATH + 'RG65.csv'
path_sim353 = config.EVAL_ROOT_PATH + 'wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt'
path_men = config.EVAL_ROOT_PATH + 'MEN/MEN_dataset_lemma_form_full'
path_simlex = config.EVAL_ROOT_PATH + 'SimLex-999.txt'
path_sim_rel = config.EVAL_ROOT_PATH + 'wordsim353_sim_rel/wordsim353_annotator1.txt'

with open(path_rg65, 'r') as reader:
    rg65 = reader.readlines()
# sim353 = open(path_sim353, 'r').readlines()
with open(path_men, 'r') as reader:
    men = reader.readlines()
men_ = list(map(lambda x: x.split(' '), men))
sim353 = open(path_sim353, 'r').readlines()
sim_rel = open(path_sim_rel, 'r').readlines()
with open( path_simlex, 'r') as reader:
    simlex = reader.readlines()

#t_total = len(dataloader) * params.batch_size # total step
VOCAB_SIZE = len(word2idx)
logging.info(f"vocab size {VOCAB_SIZE} ")

###########################################
########### Setting Model #################
###########################################

NEG_N= config.NEG_N
FN = SoftmaxLoss(VOCAB_SIZE, params.hidden_size, neg_n=NEG_N,distrib=unigram)
FN = FN.to(DEVICE)
#NCE = NCEloss(VOCAB_SIZE, params.hidden_size, DEVICE, neg_n=NEG_N,distrib=unigram)
#NCE = NCE.to(DEVICE)
optimizer = Adam(FN.parameters(),lr=params.learning_rate)

#scheduler = get_cosine_schedule_with_warmup(optimizer,4000,t_total)

def adjust_lr(optimizer, loader_len, epoch_num, global_step):
    """ Sets the learning rate to initialize LR dacayed per global step """
    # set learning rate 0.0005 -> 0.0001
    lr = params.learning_rate # lr share params.learning rate's memory id.
    sample_num = loader_len * params.batch_size
    epoch_steps = sample_num // params.batch_size
    whole_steps = epoch_steps * epoch_num
    decay_rate = 0.865

    if global_step % 100000 == 0:
        lr *= decay_rate # lr has different id by inplace operator
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

best_loss = np.inf
best_rg = -np.inf
best_men = -np.inf
global_step = 0
for epoch in range(params.epochs):
    """ one epoch : 4 train loader """
    n=9
    while n > -1:
        with open(config.SAVE_PATH + f'/features_{n}.pkl', 'rb') as r:
            features = pickle.load(r)
        logger.info(f"feature {n} loaded! ")

        dataset = Word2VecDataset(data=features)
        logger.info(len(dataset) // params.batch_size)
        dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, collate_fn=my_collate)
        loader_len = len(dataloader)
        pbar =  tqdm(dataloader)
        
        train_loss = 0.0
        step=0
        for iword, owords in pbar:
            iword, owords = iword.to(DEVICE), owords.to(DEVICE)
            #loss = NCE(iword, owords)
            loss = FN(iword, owords)
            optimizer.zero_grad()
            loss.backward()
            adjust_lr(optimizer, loader_len, params.epochs, global_step)
            nn.utils.clip_grad_norm_(FN.parameters(), params.max_grad_norm)
            optimizer.step()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            train_loss += loss.item()
            step+=1
            global_step+=1


        logger.info(f"Epoch[{epoch}/{params.epochs}] Train Loss[{float(train_loss / step):.6f}] Learning rate: {optimizer.param_groups[0]['lr']}")
        
        with torch.no_grad():
            FN.eval()
            sim_lst_rg, sim_human = cosine_sim_rg(rg65, FN, word2idx)
            logger.info(f"correlation rate RG : {spearmanr(sim_lst_rg,sim_human).correlation}")
            sim_list_men, sim_human = cosine_sim_men(men_, FN, word2idx)        
            logger.info(f"correlation rate MEN : {spearmanr(sim_list_men,sim_human).correlation})")
            sim_list_simlex, sim_human = cosine_sim_simlex(simlex, FN, word2idx)        
            logger.info(f"correlation rate SIMLEX999 : {spearmanr(sim_list_simlex,sim_human).correlation})")
    
        ################################
        ####### local eval check #######
        ################################

        n -=1
        del features

    logger.info(" One epoch training completed! ")
    with torch.no_grad():
        logger.info(" Start to evaluate Model! ")
        FN.eval()
        sim_lst_rg, sim_human = cosine_sim_rg(rg65, FN, word2idx)
        r_rg = spearmanr(sim_lst_rg,sim_human).correlation
        sim_list_men, sim_human = cosine_sim_men(men_, FN, word2idx)        
        r_men = spearmanr(sim_list_men,sim_human).correlation
        # sim_lst_353, sim_human = cosine_sim353(sim353, nce, word2idx)
        # r_353 = spearmanr(sim_lst_353,sim_human).correlation
        
    logger.info(f"r_rg :{r_rg} r_men:{r_men}")
        
    if best_loss > train_loss and r_rg > best_rg or r_men > best_men:
        best_rg = r_rg
        best_men = r_men
        best_loss = train_loss
        torch.save(FN.state_dict(), os.path.join(SAVE_DIR, '{}_{}_{}_{}_{}.pt'.format(config.MODEL_NAME,params.hidden_size, params.learning_rate, NEG_N, KEY)))
        logger.info("Model saved...")
