#-*-coding=utf-8-*-
"""Make Vocabulary and sampling distribution
"""
import os
import re
import time
import math
import pickle
import random
import logging
import multiprocessing

import heapq as heapq
from tqdm.auto import tqdm
from copy import deepcopy,copy
from collections import Counter
from collections import OrderedDict

import config

SAVE_PATH = config.SAVE_PATH_PRE

# sub-sampling code version
if config.RANDOM_DISCARD:
    threshold = 1e-03
else:
    threshold = config.THRESHOLD

MIN_THREAD=6e-09
RM_WORDS = ['\n','=']
DIVISION = 4

if config.VARIABLE_WINDOW:
    window_sizes = [3,5,7,9,11]
else:
    window_size = config.WINDOW_SIZE # this version is fixed size window size.

"""
paper : 1-sqrt(t/f)
code : f-t / f - sqrt(t/f)
The subsampling randomly discards frequent words while keeping the ranking same
"""


if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

def preprocess(data:list)-> list:
    global RM_WORDS
    new_doc=[]
    for line in data:
        line = line.strip()
        if line.startswith(RM_WORDS[1]) or line == RM_WORDS[0]:
            pass
        else:
            new_doc.extend([line])
    try:
        while True:
            new_doc.remove('')
    except:
        pass
    return new_doc

def rm(text):
    """ 
    Wiki data format, remove spearmanrcial characters.
    """
    regularExpressions = ["\u2029","\u0022","\u002C" ,"@-@" ,"$","-", "…", "·", "●", "○", "◎", "△", "▲", "◇", "■", "□", ":phone:", "☏", "※",
                          ":arrow_forward:", "▷", "ℓ", "→", "↓", "↑", "┌", "┬", "┐", "├", "┤", "┼", "─", "│", "└", "┴",
                          "┘",
                          "[0-9]+",
                          "\n|\r?\n|\r|\t",
                          "[~!@+=%^;:$]",
                          "[ ]{1,20}",
                          "\.\.|\.\.\."]
    for regularExpression in regularExpressions:
        text = re.sub(re.compile(regularExpression), " ", text)
    
    text = text.replace(',', '')
    text = text.lower()
    text = " ".join(text.split())
    return text

# make vocab
# base vocabulary - 어절단위

def get_vocab(DATA:list, min_cnt=5):
    # corpus:list
    """ corpus : doc strings in list """
    words=[]
    for doc in DATA:
        words.extend(doc.split(' '))
    
    # remove under min count 
    occurs_vocab = Counter(words)
    vocab_ = [w for w, c in occurs_vocab.items() if c >= min_cnt]
    word2idx = {w : i for i, w in enumerate(vocab_)}
    idx2word = {i:w for w, i in word2idx.items()}

    with open(os.path.join(SAVE_PATH,f'vocab{config.MIN_CNT}.pkl'), 'wb') as writer:
        pickle.dump((word2idx, idx2word),writer)
    logger.info("Vocab saved....")

    return words, word2idx, idx2word

def get_unigram_distribution(WORDS :list, min_cnt=5):
    # words : list
    # unigram distribution
    occurs_vocab = Counter(WORDS)
    unigram = [c for w, c in occurs_vocab.items() if c >= min_cnt]
    denominator = sum(unigram)
    frequency = _get_frequency(unigram, denominator)
    
    with open(os.path.join(SAVE_PATH,f'unigram{config.MIN_CNT}.pkl'), 'wb') as writer:
        pickle.dump((unigram, denominator), writer)
    logger.info("Unigram saved....")

    return unigram, frequency

def _get_frequency(unigram, denominator):
    frequency = list(map(lambda x : x / denominator, unigram))
    return frequency

def get_over_threshold( frequency, t=1e-05):
    # count frequency greater than t
    return list(filter(lambda x : x > t, frequency))

# make sentence first and then make features.
def get_features(data, frequency, word2idx, unigram, flag:bool):
    global threshold
    global window_size
    global DIVISION

    features=[]
    for doc in tqdm(data):
        doc = doc.split(' ')
        try:
            while True:
                doc.remove('(')
                doc.remove(')')
                doc.remove('.')
        except:
            pass

        if flag:
            """ random mutable window size """
            for idx in range(len(doc)):
                window_size = random.choice(window_sizes) # uniform distribution random.random()        
                try:
                    word=doc[idx]
                    if frequency[word2idx[word]] < threshold:
                        left = doc[max(idx-window_size,0):idx]
                        right = doc[idx+1:idx+(window_size+1)]
                        
                        features.append([word] + left + right)
                except Exception as e:
                    print(e)
                        
        else:
            """ sub-sampling 
                original paper&code : threshold => f-t / f - sqrt(t/f)
                each word, threshold check and then determine to add or not.
                below code, I remain target word < threshold , and other words are same.
            """
            if config.RANDOM_DISCARD:
                sub=[]    
                MAX_LEN = len(doc)
                for idx in range(len(doc)):
                    word = doc[idx]
                    freq = unigram[word2idx[word]]
                    ran = (math.sqrt(freq / (threshold * sum(unigram))) + 1) * (threshold * sum(unigram) / freq)
                    if ran < 1.: # Please need to know the ran's range.
                        continue
                    #sub.append([word] + [word])
                    # writing...

                    if len(sub) > MAX_LEN:
                        break
                    
            else:
                for idx in range(len(doc)):
                    word= doc[idx]
                    #if frequency[word2idx[word]] < threshold and frequency[word2idx[word]] > min_thread:
                    if frequency[word2idx[word]] < threshold:
                        left = doc[max(idx-window_size,0):idx]
                        right = doc[idx+1:idx+(window_size+1)]
                        features.append([word] + left + right)


# --------------------------------------------------------------------------------------------- #
# --------------------- save several files ----------------------------------------------------
    division= DIVISION # change global variable
    length = len(features)
    line = int(length / division)
    for i in range(division):
        if i == division - 1:
            with open(os.path.join(SAVE_PATH,f'samples_{config.MIN_CNT}_{i}.pkl'), 'wb') as writer:
                pickle.dump(features[line*i:], writer)
        else:
            with open(os.path.join(SAVE_PATH,f'samples_{config.MIN_CNT}_{i}.pkl'), 'wb') as writer:
                pickle.dump(features[line*i:line*(i+1)], writer)
            logger.info("Data(words) saved....")
# --------------------------------------------------------------------------------------------
    #training features
    for i in range(division):
        with open(os.path.join(SAVE_PATH,f'samples_{config.MIN_CNT}_{i}.pkl'), 'rb') as r:
            features = pickle.load(r)
        
        convert_id(features, word2idx, i)

    logging.info(" ****** All is done! ****** ")

def convert_id(features, word2idx, idx):
    indices = []
    for words in tqdm(features):
        #indices.append([word2idx[word] for word in words])
        # update version : applying min count. if word under min count appears, window skip that word.
        tmp=[]
        for word in words:
            try:
                tmp.append(word2idx[word])
            except Exception:
                pass
        indices.append(tmp)

    with open(os.path.join(SAVE_PATH,f'features_{config.MIN_CNT}_{idx}.pkl'), 'wb') as writer:
        pickle.dump(indices, writer)
    logger.info("Features saved....")

def remove_mincount(DATA, rm_words):
    DATA_=[]
    for doc in DATA:
        tmp=' '
        for word in doc.split(' '):
            if word not in rm_words:
                tmp.join(word)
        DATA_.append(tmp)
    return DATA_

def get_all_list(ROOT_PATH, FILE_LIST, break_flag=None):
    """ DATA return : List[str] 
        break flag : start to end : tuple(int)
    """
    data_ = []
    if break_flag:
        for i in tqdm(range(len(FILE_LIST))):
            if i < break_flag[1]:
                if i >= break_flag[0]:
                    with open(ROOT_PATH + FILE_LIST[i], 'r') as reader:
                        data = reader.readlines()
                    data = list(map(rm, data))
                    data = [d.replace('.','') for d in data]
                    data = preprocess(data)
                    data_ += data
                    print("sentence count : ", len(data_))
                else:
                    pass
            else:
                break

    else:
        for i in tqdm(range(len(FILE_LIST))):
            with open(ROOT_PATH + FILE_LIST[i], 'r') as reader:
                data = reader.readlines()
            data = list(map(rm, data))
            data = [d.replace('.','') for d in data]
            data = preprocess(data)
            data_ += data
            print("sentence count : ", len(data_)) 

    return data_

if __name__ == '__main__':
    # Setting info config
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
    logger =logging.getLogger(__name__)

    flag=None

    #########################
    ##### Make one file #####
    #########################
    # DATA_PATH = config.DATA_PATH
    # DATA_LIST = os.listdir(DATA_PATH)
    # DATA_LIST = sorted(DATA_LIST)

    # DATA = get_all_list(DATA_PATH, DATA_LIST, break_flag=flag)

    with open(os.path.join(config.SAVE_PATH,'data01.pkl'), 'rb') as read:
        DATA = pickle.load(read)

    # #################################
    # ##### Make needed variables #####
    # #################################
    WORDS, word2idx, _ = get_vocab(DATA, min_cnt=config.MIN_CNT)
    unigram, frequency = get_unigram_distribution(WORDS, min_cnt=config.MIN_CNT)
    get_features(DATA, frequency, word2idx, unigram, flag=config.VARIABLE_WINDOW)

    ########################################
    ### Multi processing 
    ########################################
    # num_cores= 4
    # #num_cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(num_cores)
    # manager = multiprocessing.Manager()  # syb-process has shared space

    # features=[]
    # for i in range(division):
    #     with open(os.path.join(SAVE_PATH,f'samples_{i}.pkl'), 'rb') as writer:
    #         features.append(pickle.load(writer))
    #     convert_id(feature, word2idx, i)

    # pool.map(convert_id, features)
    # pool.close()
    # pool.join()
