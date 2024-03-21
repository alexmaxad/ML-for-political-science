from mittens import GloVe 
import pandas as pd
from scipy import sparse
import json
import numpy as np
import ast
import os
import csv

from Processing.text_cleaning import *

from mittens import Mittens

def glove2dict(glove_filename):
    ''' transforms a txt file of embeddings into a dictionary
    Parameters:
    -----------
    glove_filename : embeddings txt file
    '''
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        words = []
        mats = []
        for line in reader :
            if len(clean(line[0], gram='unigram'))>0:
                words.append(clean(line[0], gram='unigram')[0])
                mats.append(np.array(list(map(float, line[1:]))))
    embed = {words[i]: mats[i] for i in range(len(words))}
    return embed