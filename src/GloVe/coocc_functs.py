from __future__ import division
from collections import Counter
import operator
import random
import json
import numpy as np
from time import sleep
from dask import dataframe as dd
import ast
from scipy import sparse
from random import random
from time import sleep
from scipy.sparse import dok_matrix
from multiprocessing import Pool
import os
from functools import partial,reduce
from operator import add

def vocab_dic(fichier) :
    '''  Returns
    vocab : the entire list of words that are used, without doubles
    word2idx : dictionary created with this vocabulary
    
    Parameters:
    -----------
    fichier : the words
    '''
    with open(fichier) as f:
        vocab_inter = json.load(f)
    vocab = list(set(vocab_inter))
    word2idx = {v: i for i, v in enumerate(vocab)}
    return vocab, word2idx

def inter_coocc(items, word2idx):
        '''
        Creates the cooccurence matrix for the words in the items

        Parameters:
        -----------
        items : text to process
        '''
        print('dans inter_coocc')
        coocc = dok_matrix((len(word2idx), len(word2idx))) 
        #loop on subItems
        for (j,t) in items:
          word_counts = Counter(t)
          window = list(word_counts.items())
          for i, (word, count1) in enumerate(window):
            for (context, count2) in window[i:i+10]:
                try :
                    coocc[word2idx[word], word2idx[context]] += count1 * count2
                    if context != word:
                        coocc[word2idx[context], word2idx[word]] += count1 * count2
                except : 
                     coocc = coocc
          if j % 10000 == 0:
                print(t)
                print(j)
        return(coocc)

def split(a, n):
    ''' Function to split a list in n evenly subpart'''
    k, m = divmod(len(a), n)
    return ([a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)])