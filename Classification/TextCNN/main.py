import os 
import re
import time
import json
import pickle
import logging
import numpy as np

from gensim.models import Word2Vec, KeyedVectors

# torchtext version >= 0.6.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adadelta

from tqdm.auto import tqdm
from torchtext import data
from torchtext.data import Batch, Dataset, Iterator
from torchtext.data import Field, LabelField
from torchtext.datasets import SST,TREC

from src.model.config import ModelConfig, Struct
from src.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file # settting logging module

SEED = 42
DATA = 'TREC'
CUDA = False
DEBUG = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODE = 'static' # nonstatic[]
WORD_VECTORS = 'rand' # choices = [rand, word2vec]
DIM = 300

#  set seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) # if use multi-GPU

LOGGER.info(MODE)
LOGGER.info(WORD_VECTORS)
#########################
# Custom class Setting
#########################
class GSST(SST):
    urls = ['http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip']
    dirname = 'trees'
    name = 'sst'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, subtrees=False,
                 fine_grained=True, **kwargs):
        fields = [('text', text_field), ('label', label_field)]

        def get_label_str(label):
            pre = 'very ' if fine_grained else ''
            return {'0': pre + 'negative', '1': 'negative', '2': 'neutral',
                    '3': 'positive', '4': pre + 'positive', None: None}[label]
        label_field.preprocessing = data.Pipeline(get_label_str)
        with open(os.path.expanduser(path)) as f:
            if subtrees:
                examples = [ex for line in f for ex in
                            data.Example.fromtree(line, fields, True)]
            else:
                examples = [data.Example.fromtree(line, fields) for line in f]
        super(SST, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='train.txt', validation='dev.txt', test='test.txt',
               train_subtrees=False, **kwargs):

        path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), text_field, label_field, subtrees=train_subtrees,
            **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), text_field, label_field, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), text_field, label_field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, shuffle=True, device=device)


#########################
# Model Setting
#########################
# Implementation CNN-rand : all words are randomly initialized and then modified during training.
# torch version

class _ActivationLambda(nn.Module):
    """Wrapper around non PyTorch, lambda based activations to display them as modules whenever printing model."""

    def __init__(self, func, name: str):
        super().__init__()
        self._name = name
        self._func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._func(x)

    def _get_name(self):
        return self._name
    
class TextCNN(nn.Module):
    """
    self.criterion  : output layer for logistic regression, projecting data points onto a set of hyperplanes, 
                                            the distance to which is used to determine a class membership probability.
    """
    def __init__(self, channel_in, channel_out, filter_size, non_linear='relu', p=0.2):
        super(TextCNN, self).__init__()
        self.conv_layers = nn.ModuleList(nn.Conv1d(channel_in, channel_out, kernel_size=filter_size) for filter_size in cfg.filter_hs)
        if non_linear == 'tanh':
            self.act = nn.Tanh()
        elif non_linear == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = _ActivationLambda(lambda x: x, "Ident") # usage : self.act(x)
                
        self.dropout = nn.Dropout(p=p)
        self.hidden0 = nn.Sequential(nn.Linear(cfg.hidden_units[0]*3, cfg.hidden_units[0]), self.act) # 300,classes
        self.hidden1 = nn.Sequential(nn.Linear(cfg.hidden_units[0]*3, cfg.hidden_units[0]), self.act)
        self.classifier = nn.Linear(cfg.hidden_units[0], cfg.hidden_units[1])
        self.criterion = nn.CrossEntropyLoss() # log softmax
        
        
    def forward(self, indicies, labels):
        """ indicies : batch size, channels, sequence """
        outputs = []
        for conv_layer in self.conv_layers:
            feature_maps = conv_layer(indicies)
            conv_out = self.act(feature_maps)
            conv_pool_out = F.max_pool1d(conv_out, conv_out.shape[-1]) # max-pooling
            output = conv_pool_out.flatten(1) # B,H,1
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=1) # B,H
        #x = outputs
        outputs = self.hidden0(outputs)
        outputs = self.dropout(outputs) # dropout input
        preds = self.classifier(outputs)
        #resid = self.hidden1(x)
        #preds = self.classifier(outputs + resid)
        # get argmax for Prec@1
        
        loss = self.criterion(preds, labels) # errors
        return loss, preds.argmax(-1)

def add_unknown_words(word_vecs, vocab, min_df=0, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def load_bin_vec(W2V:object, VOCAB : object):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec, unknown vectors -> random vector in.
    """
    idx2word = {w:i for i, w in enumerate(W2V.index2word)}
    
    word_vecs = {}
    for word, idx in tqdm(VOCAB.vocab.stoi.items()):
        if word in W2V.index2word:
            index = idx2word[word]
            word_vecs[word] =  W2V.vectors[index]
        else:
            add_unknown_words(word_vecs, VOCAB.vocab.stoi)
            
    return word_vecs

    
if __name__ == '__main__':
    # generate key
    key = time.asctime()[11:-5]
    # load json object
    with open('src/model/params.json', 'r') as json_file:
        jf = json.load(json_file)
        
    cfg = ModelConfig.from_dict(jf)
    cfg = Struct(cfg.hyperparams)
    
    ###############################
    ## Check Configuration
    ###############################
    LOGGER.info(cfg)
    
    if DATA == 'SST':
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, val, test = GSST.splits(TEXT, LABEL, root='./data')

        # build vocabulary
        TEXT.build_vocab(train)
        LABEL.build_vocab(train)

        # define loader(iter class)
        loaders = GSST.iters(batch_size=cfg.batch_size, device=DEVICE if CUDA else "cpu", vectors=None)

        # split loader
        train_loader, eval_loader, test_loader = loaders

    elif DATA == 'TREC':

        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, test = TREC.splits(TEXT, LABEL, root='./data')

        # build vocabulary
        TEXT.build_vocab(train)
        LABEL.build_vocab(train)

        # define loader(iter class)
        loaders = TREC.iters(batch_size=cfg.batch_size, device=DEVICE if CUDA else "cpu", vectors=None)

        # split loader
        train_loader, test_loader = loaders

        #TEXT.vocab.itos
        LABEL.vocab.itos
    else:
        print("choose data ! SST, TREC")
    
    ###############################
    ## Set Variable
    ###############################
    if WORD_VECTORS == 'rand':
        # random initialize all word vectors
        W1 = torch.empty(len(TEXT.vocab.itos), DIM).uniform_(-0.5, 0.5)
        # do not train  
        W1[1,:]=0
        U = W1
    else:
        W2 = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
        w2v = load_bin_vec(W2, TEXT) # return dict
        U = np.vstack(list(w2v.values())) # list(iterable)
        U = torch.Tensor(U)

    # set the trainable parameter or not.
    if MODE == 'static':
        # static : pre-trained vectors are kept
        pass
    else:
        # MODE == 'nonstatic'
        if isinstance(U, torch.Tensor):
            U = nn.Parameter(U)
        else:
            LOGGER.info("Type error, U should be torch.Tensor. ")
            
    model = TextCNN(cfg.hidden_units[0] * 3,cfg.hidden_units[0],cfg.filter_hs)
    model.train()

    LR = 1e-04
    #optimizer = Adam(model.parameters(), lr=LR) # 0.001
    optimizer = Adadelta(model.parameters(), lr=1.)
    
    ###################################
    ### Training
    ###################################
    best_loss = float('inf')
    best_acc = float('-inf')
    for epoch in range(cfg.n_epochs):
        epoch_loss=0.0
        steps=0
        num=0
        pbar = tqdm(train_loader)
        pbar_eval = tqdm(test_loader)
        for field in pbar:
            # check shape
            if DEBUG:
                S, B = field.text.shape
                print(f"field text shape is Sequence : {S},Batch : {B}")

            if CUDA:
                U = U.to(DEVICE)
                field.text= (lambda x : x.to(DEVICE))(field.text)
                field.label = (lambda x : x.to(DEVICE))(field.label)
                model = model.to(DEVICE)

            field.text = field.text.transpose(-1,0)
            labels = field.label-1
            indicies = U[field.text]
            indicies = indicies.permute(0,2,1)

            optimizer.zero_grad()
            loss, train_preds = model(indicies, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            steps+=1
            epoch_loss += loss.item()

            with torch.no_grad():
                num += (train_preds == labels).int().sum()

            avg_loss = epoch_loss / steps
        print(f"{epoch}/{cfg.n_epochs}  AvgLoss :{avg_loss}")
        avg_acc = num / (len(train_loader) * cfg.batch_size)
        print(f"Epoch[{epoch}/{cfg.n_epochs}] Train Accuracy : {avg_acc*100:.2f}")

        eval_num=0
        model.eval()
        cnt=0
        for field in pbar_eval:
            field.text = field.text.transpose(-1,0)
            if field.text.shape[-1] < 5:
                continue
                
            labels = field.label-1
            indicies = U[field.text]
            indicies = indicies.permute(0,2,1)
            cnt+=1

            with torch.no_grad():
                eval_loss, preds = model(indicies, labels)
                eval_num += (preds == labels).int().sum()
                
        eval_avg_acc = eval_num / (cnt* cfg.batch_size)        
#         eval_avg_acc = eval_num / (len(test_loader) * cfg.batch_size)
        print(f"Epoch[{epoch}/{cfg.n_epochs}] Eval Accuracy : {eval_avg_acc*100:.2f}")
    
        if best_loss > avg_loss and best_acc < eval_avg_acc:
            best_loss = loss
            best_acc = eval_avg_acc
            torch.save(model.state_dict(), os.path.join('result/', '{}_{}_{}_{}_{}.pt'.format('textcnn', LR, WORD_VECTORS, MODE, key)))
            print("Model saved...")







