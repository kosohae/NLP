# -*- coding: utf-8 -*-
import logging
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,BatchSampler, WeightedRandomSampler

from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import LambdaLR,CosineAnnealingLR 

#############################################
#### LOADER  ####
#############################################

class Word2VecDataset(Dataset):
    """ data to array, tensorize """
    def __init__(self, data_pth=None, data=None):
        super().__init__()
        if data is None:
            self.data = pickle.load(open(data_pth, 'rb'))
        else:
            self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        iword, owords = self.data[idx][0], self.data[idx][1:]
        return torch.LongTensor([iword]), torch.LongTensor(owords)

def my_collate(batch):
    #length = torch.tensor([ torch.tensor(words[1].shape[0]) for words in batch])
    batch_iwords = torch.LongTensor([words[0] for words in batch])
    batch_owords = [torch.LongTensor(words[1]) for words in batch]
    batch_owords = torch.nn.utils.rnn.pad_sequence(batch_owords)
    ## compute mask
    #mask = (batch_owords != 0).to(device)
    return batch_iwords, batch_owords


class SubwordDataset(Dataset):
    """ data to array, tensorize """
    def __init__(self, data_pth=None, data=None):
        super().__init__()
        if data is None:
            self.data = pickle.load(open(data_pth, 'rb'))
        else:
            self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        iword, owords = self.data[idx][0], self.data[idx][1]
        return torch.LongTensor(iword), torch.LongTensor(owords)
    
def collate_subword(batch):
    batch_iwords = [iwords for iwords, owords in batch]
    batch_owords = [owords for iwords, owords in batch]
    
    batch_iwords = torch.nn.utils.rnn.pad_sequence(batch_iwords)
    batch_owords = torch.nn.utils.rnn.pad_sequence(batch_owords)
    return batch_iwords, batch_owords

class Node:
    def __init__(self, word=None):
        self.word = word
        self.count = 0
        self.path = None # from root to the word (leaf), list of the indices
        self.code = None # Huffman encoding 

class HSDataset(Dataset):
    """ vocab word 마다의 path, code 저장 """
    def __init__(self, node_pth, data_pth=None, vocab_pth=None, data=None):
        super().__init__()
        if data is None:
            with open(data_pth, 'rb') as r:
                self.data = pickle.load(r)
        else:
            self.data = data
            
        with open(node_pth, 'rb') as r:
            self.nodes = pickle.load(r)
            
        self.w2i, self.i2w = pickle.load(open(vocab_pth, 'rb'))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        token, context_words = self.data[idx][0], self.data[idx][1:]
        # one token's paths and code
        context_batch=[]
        paths_batch=[]
        labels_batch=[]

        paths = self.nodes[token].path # [213,436,213,...] # K
        labels = self.nodes[token].code # [0,1,0,...]
        
        if len(paths)  > 0 and len(labels) > 0 and len(context_words) > 0:
            # one token have window size's context words # K
            for context in context_words: # C
                context_batch.append(context) # 1, C
                paths_batch.append(paths) # 1,C,K
                labels_batch.append(labels) # 1,C,K
        
        return torch.LongTensor(context_batch), torch.LongTensor(paths_batch), torch.LongTensor(labels_batch)

#############################################
#### Skip-grram with subword information ####
#############################################
class SubInfoNCEloss(nn.Module):
    def __init__(self, vocab_size, embed_dim, neg_n=10, power=0.75, distrib=None, padding_idx=0):
        """ 
        Args : weights list(vocab_size)
        """
        super(SubInfoNCEloss, self).__init__()
        self.neg_n = neg_n
        self.vocab_size = vocab_size
        self.embedding_i = nn.Embedding(vocab_size, embed_dim,padding_idx=0) # vocab size, hidden size
        self.embedding_os = nn.Embedding(vocab_size, embed_dim,padding_idx=0)
        
        self.embed_dim = embed_dim
        self.power = power
        self.distrib = distrib
        self.initialize()
        
    def initialize(self):
        """
        init the weight as original word2vec do.
        :return: None
        """
        initrange = 0.5 / self.embed_dim
        self.embedding_i.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embed_dim), torch.FloatTensor(self.vocab_size - 1, self.embed_dim).uniform_(-initrange, initrange)]))
        self.embedding_os.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embed_dim), torch.FloatTensor(self.vocab_size - 1, self.embed_dim).uniform_(-initrange, initrange)]))
        self.embedding_i.weight.requires_grad=True
        self.embedding_os.weight.requires_grad=True
                                                          
    def forward(self, i_words, o_words): # obj generate 후, call함.
        """
        i_word: in word 
        o_words: out words
        """
        batch_size = i_words.shape[1]
        context_size = o_words.shape[0]
        device = i_words.device
        
        # paper, Negative sampling distribution : proposed distribution U(w)^3/4 /Z
        if self.distrib is not None:
            wt = torch.pow(torch.FloatTensor(self.distrib), self.power)
            wt /= wt.sum()
            wt = wt.to(device)
            
            # make uniform distribution changed 
#             wt[i_words] = 0
#             wt[o_words] = 0
            
        n_words = torch.multinomial(wt ,batch_size * context_size * self.neg_n, replacement=True).view(batch_size, -1)
        
        # weights uniform distribution
        if self.distrib is None:
            n_words = torch.empty(batch_size *context_size * self.neg_n).uniform_(0, self.vocab_size-1).long()
       
        # i vectors => sub vectors
        i_words = i_words.permute(1,0) # B,K
        o_words = o_words.permute(1,0) # B,C
        
        i_vec = self.embedding_i(i_words) # B,K,D
        o_vec = self.embedding_os(o_words) # B, C-1, D
        o_vec_n = self.embedding_os(n_words).neg() # B, N, D

        loss_pos = torch.bmm(o_vec, i_vec.permute(0,2,1)).sum(-1).squeeze().sigmoid().log().mean(-1)# B,C
        loss_neg = torch.bmm(o_vec_n, i_vec.permute(0,2,1)).sum(-1).squeeze().sigmoid().log().view(-1, context_size, self.neg_n).mean(-1)

        loss = torch.sum(loss_pos) + torch.sum(loss_neg)  # batch sum

        return -loss  # optimizer minimize loss

#############################################
###### Skip-Gram Model with NCE loss ########
#############################################

class HSsoftmaxLoss(nn.Module):
    def __init__(self, vocab_size, embed_dim, DEVICE=None, distrib=None):
        """ 
        Args:
        vocab_size : word2idx size
        embed_dim : hidden size (projection layer)
        DEVICE : gpu device index
        neg_n : negative sampling number
        power : ease negative sampling distribution
        distrib : unigram distribution
        padding_idx : None
        """
        super(HSsoftmaxLoss, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_syn0 = nn.Embedding(vocab_size, embed_dim) # vocab size, hidden size
        self.embedding_syn1 = nn.Embedding(vocab_size, embed_dim)
     
        self.embed_dim = embed_dim
        self.distrib = distrib
        self.initialize()
        
    def initialize(self):
        """
        init the weight as original word2vec do.
        :return: None
        """
        initrange = 0.5 / self.embed_dim
        self.embedding_syn0.weight = nn.Parameter(torch.FloatTensor(self.vocab_size - 1, self.embed_dim).uniform_(-initrange, initrange))
        self.embedding_syn1.weight = nn.Parameter(torch.FloatTensor(self.vocab_size - 1, self.embed_dim).uniform_(-initrange, initrange))
                                                          
    def forward(self, c_words, paths, labels): # obj generate 후, call함.
        """
        c_words: context words by one token 
        paths : paths by one token
        labels : codes by one token
        Training one token hierarchy softmax
        """
        #context_size = c_words.shape[-1] # C
        # No batch style
        c_words.squeeze_() # C,
        paths.squeeze_() # C, K (K is path from one token)
        labels.squeeze_() # C, K 
        
        c_vec = self.embedding_syn0(c_words) # C, D
        path_vec = self.embedding_syn1(paths) # C, K, D
        # maximize the average log probability
        z = torch.matmul(c_vec, path_vec[0].T).sigmoid().log() #  C x D dot D x K  => C, K
        loss = F.binary_cross_entropy(z, labels, reduction='sum')
        #loss = (labels - z).sum() # minimize loss
        return loss
    
class SoftmaxLoss(nn.Module):
    def __init__(self, vocab_size, embed_dim, neg_n=10, power=0.75, distrib=None):
        """ 
        Args:
        vocab_size : word2idx size
        embed_dim : hidden size (projection layer)
        power : ease negative sampling distribution
        distrib : unigram distribution
        padding_idx : 
        """
        super(SoftmaxLoss, self).__init__()
        self.neg_n = neg_n
        self.vocab_size = vocab_size
        self.embedding_i = nn.Embedding(vocab_size, embed_dim,padding_idx=0) # vocab size, hidden size
        self.embedding_os = nn.Embedding(vocab_size, embed_dim,padding_idx=0)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        #self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        self.embed_dim = embed_dim
        self.power = power
        self.distrib = distrib
        self.initialize()
    
    def custom_cross_entropy(self, logits, labels):
        """ element-wise product and summation. """
        loss = torch.sum(logits * labels) / logits.shape[0]
        return loss
    
    def initialize(self):
        """
        init the weight as original word2vec do.
        :return: None
        """
        # initrange = 0.5 / self.embed_dim
        initrange = 0.5 / self.embed_dim
        self.embedding_i.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embed_dim), torch.FloatTensor(self.vocab_size - 1, self.embed_dim).uniform_(-initrange, initrange)]))
        self.embedding_os.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embed_dim), torch.FloatTensor(self.vocab_size - 1, self.embed_dim).uniform_(-initrange, initrange)]))
        self.embedding_i.weight.requires_grad=True
        self.embedding_os.weight.requires_grad=True
        
    def lookup_table(self, owords, device):
        """ transform index to one-hot vector """
        #one_hot = torch.eye(self.vocab_size, self.vocab_size)
        #owords_onehot = one_hot[owords].sum(dim=1)
        y = torch.LongTensor(owords.shape[0], owords.shape[-1]).random_() % self.vocab_size
        y = y.to(device)
        owords_onehot = torch.FloatTensor(owords.shape[0], self.vocab_size).to(device)
        owords_onehot.zero_()
        owords_onehot.scatter_(1, y, 1)
        return owords_onehot
    
    def forward(self, i_word, o_words): # obj generate 후, call함.
        """
        i_word: in word 
        o_words: out words
        """
        device = i_word.device
        batch_size = i_word.shape[0]
        context_size = o_words.shape[0]
             
        # version : 
        o_words = o_words.permute(1,0) # B,C
        i_vec = self.embedding_i(i_word) # B,D
        o_vec = self.embedding_os(o_words) # B, C-1, D
#         o_vec_n = self.embedding_os(n_words).neg() # B, N, D
        
        output = torch.matmul(i_vec, self.embedding_i.weight.permute(1,0)) # B, V
        logits = self.logsoftmax(output)
        labels = self.lookup_table(o_words, device)
        labels = labels.to(device)
        loss = self.custom_cross_entropy(logits, labels)
        #print(self.embedding_i.weight.grad)
#         loss_neg = torch.bmm(o_vec_n, i_vec.permute(0,2,1)).squeeze().sigmoid().log().view(-1, context_size, self.neg_n).mean(-1)#.sum(-1)
        print(loss)
        return -loss  # optimizer minimize loss

        
class NCEloss(nn.Module):
    def __init__(self, vocab_size, embed_dim, neg_n=10, power=0.75, distrib=None):
        """ 
        Args:
        vocab_size : word2idx size
        embed_dim : hidden size (projection layer)
        DEVICE : gpu device index
        neg_n : negative sampling number
        power : ease negative sampling distribution
        distrib : unigram distribution
        padding_idx : 
        """
        super(NCEloss, self).__init__()
        self.neg_n = neg_n
        self.vocab_size = vocab_size
        self.embedding_i = nn.Embedding(vocab_size, embed_dim,padding_idx=0) # vocab size, hidden size
        self.embedding_os = nn.Embedding(vocab_size, embed_dim,padding_idx=0)
        
        self.embed_dim = embed_dim
        self.power = power
        self.distrib = distrib
        self.initialize()
        
    def initialize(self):
        """
        init the weight as original word2vec do.
        :return: None
        """
        # initrange = 0.5 / self.embed_dim
        initrange = 0.05 / self.embed_dim
        self.embedding_i.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embed_dim), torch.FloatTensor(self.vocab_size - 1, self.embed_dim).uniform_(-initrange, initrange)]))
        self.embedding_os.weight = nn.Parameter(torch.cat([torch.zeros(1, self.embed_dim), torch.FloatTensor(self.vocab_size - 1, self.embed_dim).uniform_(-initrange, initrange)]))
        self.embedding_i.weight.requires_grad=True
        self.embedding_os.weight.requires_grad=True
                                                          
    def forward(self, i_word, o_words): # obj generate 후, call함.
        """
        i_word: in word 
        o_words: out words
        """
        batch_size = i_word.shape[0]
        context_size = o_words.shape[0]
        device = i_word.device
        
        # paper, Negative sampling distribution : proposed distribution U(w)^3/4 /Z
        if self.distrib is not None:  # make distribution more soft, high frequency down.
            wt = torch.pow(torch.FloatTensor(self.distrib), self.power)
            wt /= wt.sum()
            wt = wt.to(device)
            # make uniform distribution changed - add idea
            wt[i_word] = 0
            wt[o_words] = 0
            
        n_words = torch.multinomial(wt ,batch_size * context_size * self.neg_n, replacement=True).view(batch_size, -1)
        
        # weights uniform distribution
        if self.distrib is None:
            n_words = torch.empty(batch_size *context_size * self.neg_n).uniform_(0, self.vocab_size-1).long()
        
        # version : 
        o_words = o_words.permute(1,0) # B,C
        i_vec = self.embedding_i(i_word).unsqueeze(1) # B,1,D
        o_vec = self.embedding_os(o_words) # B, C-1, D
        o_vec_n = self.embedding_os(n_words).neg() # B, N, D
        
        ###################################
        # paper version => sigmoid
        # ablation other activation function
        ###################################
        loss_pos = torch.bmm(o_vec, i_vec.permute(0,2,1)).squeeze().tanh().log().mean(-1)#.sum(-1) # context sum
        loss_neg = torch.bmm(o_vec_n, i_vec.permute(0,2,1)).squeeze().tanh().log().view(-1, context_size, self.neg_n).mean(-1)
        
#         loss_pos = torch.bmm(o_vec, i_vec.permute(0,2,1)).squeeze().sigmoid().log().mean(-1)#.sum(-1) # context sum
#         loss_neg = torch.bmm(o_vec_n, i_vec.permute(0,2,1)).squeeze().sigmoid().log().view(-1, context_size, self.neg_n).mean(-1)#.sum(-1)
        
        loss = torch.sum(loss_pos) + torch.sum(loss_neg)  # batch sum
        return -loss  # optimizer minimize loss

if __name__ == '__main__':
    # Set Logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
    logger =logging.getLogger(__name__)
    
    # Seed
    SEED = 42 # predictable random variables
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Test Model


