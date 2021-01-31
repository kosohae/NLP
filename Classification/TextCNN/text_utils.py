# -*- coding: utf-8 -*-
import re

def remove_special(text):
    """ 
    remove spearmanrcial characters.
    """
    regularExpressions = ["\u2029","\u0022","\u002C" ,"@-@" ,"-", "…", "·", "●", "○", "◎", "△", "▲", "◇", "■", "□", ":phone:", "☏", "※",
                          ":arrow_forward:", "▷", "ℓ", "→", "↓", "↑", "┌", "┬", "┐", "├", "┤", "┼", "─", "│", "└", "┴",
                          "┘",
                          "\n|\r?\n|\r|\t",
                          '[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]',
                          '[\!@+=%^*;:]',
                          "[ ]{1,20}",
                          "\.\.|\.\.\."]
    for regularExpression in regularExpressions:
        text = re.sub(re.compile(regularExpression), " ", text)

    text = text.replace(',', '')
    return text

def postprocess(context:str) -> str:
    context = context.replace('-','')
    context = context.replace('*','')
    postprocessed = context.replace('u005C','')
    return postprocessed

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()
