from __future__ import division
import numpy as np
from collections import defaultdict
import math
import json
from Polarization.polarization_functions import *

def get_counts(text, vocab):
    counts = {w: 0 for w in vocab}
    for split in text:
        count = 0
        prev = ''
        for w in split:
            if w == '':
                continue
            if w in vocab:
                counts[w] += 1
            if count > 0:
                bigram = prev + ' ' + w
                if bigram in vocab:
                    counts[bigram] += 1
            count += 1
            prev = w
    return counts

def log_odds(counts1, counts2, prior, zscore = True):
    # code from Dan Jurafsky
    # note: counts1 will be positive and counts2 will be negative

    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    n1 = sum(counts1.values())
    n2 = sum(counts2.values())

    # since we use the sum of counts from the two groups as a prior, this is equivalent to a simple log odds ratio
    nprior = sum(prior.values())
    for word in prior.keys():
        if prior[word] == 0:
            delta[word] = 0
            continue
        l1 = float(counts1[word] + prior[word]) / (( n1 + nprior ) - (counts1[word] + prior[word]))
        l2 = float(counts2[word] + prior[word]) / (( n2 + nprior ) - (counts2[word] + prior[word]))
        sigmasquared[word] = 1/(float(counts1[word]) + float(prior[word])) + 1/(float(counts2[word]) + float(prior[word]))
        sigma[word] = math.sqrt(sigmasquared[word])
        delta[word] = (math.log(l1) - math.log(l2))
        if zscore:
            delta[word] /= sigma[word]
    return delta

def get_log_odds_values(df_speeches, words2idx, party_1, party_2):

    dem_tweets, rep_tweets = df_speeches[df_speeches['party'] == party_1], df_speeches[df_speeches['party'] == party_2]

    # get counts
    counts1 = get_counts(rep_tweets['text'], words2idx)
    counts2 = get_counts(dem_tweets['text'], words2idx)
    prior = {}
    for k, v in counts1.items():
        prior[k] = v + counts2[k]

    # get log odds
    # note: we don't z-score because that makes the absolute values for large events significantly smaller than for smaller
    # events. however, z-scoring doesn't make a difference for our results, since we simply look at whether the log odds
    # are negative or positive (rather than their absolute value)
    delta = log_odds(counts1, counts2, prior, False)
    return prior, counts1, counts2, delta

def from_unigrams_to_bigrams(list_of_strings):
    return [list_of_strings[i]+' '+list_of_strings[i+1] for i in range(len(list_of_strings)-1)]

def get_word_partisanship(df_speeches, year, party_1, party_2, bigram=False):

    with open('data/vocabs/vocab_'+str(year)+'.json') as f:
        vocab = json.load(f)

    if bigram :
        df_speeches['text'] = df_speeches['text'].apply(from_unigrams_to_bigrams)
        vocab = [vocab[i]+' '+vocab[i+1] for i in range(len(vocab)-1)]

    words2idx = {w: i for i, w in enumerate(vocab)}
    idx2words = {i:w for i, w in enumerate(vocab)}

    # get log odds
    prior, counts1, counts2, delta = get_log_odds_values(df_speeches, words2idx, party_1, party_2)

    # get counts for posterior, mutual information and chi square
    dem_tweets, rep_tweets = df_speeches[df_speeches['party'] == 'Lab'], df_speeches[df_speeches['party'] == 'Con']
    dem_counts = get_user_token_counts(dem_tweets, words2idx)
    rep_counts = get_user_token_counts(rep_tweets, words2idx)
    dem_nonzero = set(dem_counts.nonzero()[0])
    rep_nonzero = set(rep_counts.nonzero()[0])
    dem_counts = dem_counts[np.array([(i in dem_nonzero) for i in range(dem_counts.shape[0])]),
                 :]  # filter users who did not use words from vocab
    rep_counts = rep_counts[np.array([(i in rep_nonzero) for i in range(rep_counts.shape[0])]), :]

    # calculate posterior
    dem_q = get_party_q(dem_counts)
    rep_q = get_party_q(rep_counts)
    token_scores_rep = get_rho(dem_q, rep_q)

    # mutual information and chi square
    dem_no = dem_counts.shape[0]
    rep_no = rep_counts.shape[0]
    dem_t = get_token_user_counts(dem_counts)
    rep_t = get_token_user_counts(rep_counts)
    dem_not_t = dem_no - dem_t + 2  # because of add one smoothing
    rep_not_t = rep_no - rep_t + 2  # because of add one smoothing
    mutual_info = mutual_information(dem_t, rep_t, dem_not_t, rep_not_t, dem_no, rep_no)
    chi = chi_square(dem_t, rep_t, dem_not_t, rep_not_t, dem_no, rep_no)


    features = np.ndarray((8, len(vocab)))  # prior, rep_count, dem_count, log odds, posterior, mutual information, chi square
    for w in vocab:
        features[0, words2idx[w]] = prior[w]
        features[1, words2idx[w]] = counts1[w]
        features[2, words2idx[w]] = counts2[w]
        features[3, words2idx[w]] = delta[w]
        features[4, words2idx[w]] = token_scores_rep[words2idx[w]]
        features[5, words2idx[w]] = mutual_info[words2idx[w]]
        features[6, words2idx[w]] = chi[words2idx[w]]

    return features, idx2words, words2idx