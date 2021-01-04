# -*- coding: utf-8 -*-
"""
Utils for math & statistics
"""
import re
import pickle
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import pairwise_distances

class WordUtils:
    def __init__(self, vocab, model=None):
        if model:
            self.vectors = model.embedding_i.weight.detach().numpy()
        self.word2idx, self.idx2word = vocab
        
    def nearest_neighbors(self, word, k=1, exclude=[], metric="cosine"):
        """
        Find nearest neighbor of given word.
        Nearest neighbors.
        """
        if isinstance(word, str):
            assert word in self.word2idx, "Word not found in the vocabulary"
            v = self.vectors[self.word2idx[word]]
        else: # word is vector 
            v = word

        # distance between [V, H] and [1, H]
        D = pairwise_distances(self.vectors, v.reshape(1, -1), metric=metric) 

        if isinstance(word, str):
            D[self.word2idx[word]] = D.max()

        for w in exclude:
            D[self.word2idx[w]] = D.max()

        return [self.idx2word[id] for id in D.argsort(axis=0).flatten()[0:k]]

def clip_grads(grads, max_norm):
    total_norm=0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm - 1e-06)
    if rate < 1:
        for grad in grads:
            grad*=rate # inplace operator


def cosine_similarity(x, y)-> int: 
    """
    1D tensor input 
    evalution related paper : https://arxiv.org/pdf/1901.09785.pdf
    """
    return x.dot(y) / (norm(x)* norm(y)) # l2 norm in paper

def pearsonr(x, y):
    """
    Calculates a Pearson correlation coefficient and the p-value for testing
    non-correlation.
    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear
    relationship. Positive correlations imply that as x increases, so does
    y. Negative correlations imply that as x increases, y decreases.
    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.
    Parameters
    ----------
    x : (N,) array_like
        Input
    y : (N,) array_like
        Input
    Returns
    -------
    (Pearson's correlation coefficient,
     2-tailed p-value)
    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation
    """
    # x and y should have same length.
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x-mx, y-my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(ss(xm) * ss(ym))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)
    df = n-2
    if abs(r) == 1.0:
        prob = 0.0
    else:
        t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
        prob = betai(0.5*df, 0.5, df / (df + t_squared))
    return r, prob
