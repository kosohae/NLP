import os

#####################
# Default-config
#####################
DATA_PATH = '../dataset/wrd/google_one_billion_monolingual_tokenized_shuffled/'
SAVE_PATH = 'results'
SAVE_PATH_PRE = 'results/pkl'
EVAL_ROOT_PATH = 'data/eval/'
files = os.listdir(EVAL_ROOT_PATH)
MODEL_NAME = 'sg_sf'
THRESHOLD = 1e-03
RANDOM_DISCARD = False
WINDOW_SIZE = 7 # default is 5
VARIABLE_WINDOW = True
NEG_N = 15
SEED = 42 # predictable random variables
MIN_CNT=5
#####################
# Hyper-Params
#####################


