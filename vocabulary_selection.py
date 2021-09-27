#        Interpreting Node Embedding with Text-Labeled Graphs
	  
#   File:     vocabulary_selection.py 
#   Authors:  Giuseppe Serra - giuseppe.serra@neclab.eu | gxs824@student.bham.ac.uk
#             Zhao Xu - zhao.xu@neclab.eu

# NEC Laboratories Europe GmbH, Copyright (c) <year>, All rights reserved.  

#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
 
#        PROPRIETARY INFORMATION ---  

# SOFTWARE LICENSE AGREEMENT

# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.

# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor. 

# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).

# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.

# COPYRIGHT: The Software is owned by Licensor.  

# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.

# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.

# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.

# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.

# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.

# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.

# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.  

# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.

# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.

# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.

# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.  

# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.

# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.

# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.

# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.

# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.

# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.

#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.


# -*- coding: utf-8 -*-
import os
import math
import pickle
import enchant
import matplotlib
matplotlib.use('Agg')
import numpy as np
from collections import Counter
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import utils

d = enchant.Dict('EN')
DIR_DATA = utils.DIR_DATA
DIR_PREPROCESSED_DATA = utils.DIR_PREPROCESSED_DATA
MODEL_PATH = utils.MODEL_PATH
DIR_TRAIN_TEST_DATA = utils.DIR_TRAIN_TEST_DATA

model = Word2Vec.load(MODEL_PATH)
model.init_sims()  # to get L2-normalized word vectors
rank = model.wv.index2word
words_set = set(rank)

def compute_df_tf(corpus):
    '''
    INPUT
    ----------
    corpus : {doc_ID : document text}

    OUTPUT
    ----------
    df : {word : number of documents where the current word occurs}
    tf : {doc_ID : {word : number of times the word occurs in the current document}}
    '''

    df_out = Counter()
    tf_out = {}

    for ID, doc in corpus.iteritems():

        tf_out[ID] = [Counter(doc), len(doc)]

        for word in set(doc):
            df_out[word] += 1

    return df_out, tf_out


def compute_idf(df_in, num_docs):
    '''
    INPUT
    ----------
    df : {word : number of documents where the current word occurs}
    N_docs : number of documents in the corpus

    OUTPUT
    ----------
    idf : {word : inverse document frequency}
    '''

    idf = {}

    for word in df_in.keys():
        idf[word] = math.log(num_docs/(1.0+df_in[word]))

    return idf


def compute_tf_idf(tf_in, idf_in, trained_words=words_set):
    '''
    INPUT
    ----------
    tf : {doc_ID : {word : number of times the word occurs in the current document}}
    idf : {word : inverse document frequency}

    OUTPUT
    ----------
    tf_idf : {doc_ID : {word : tf_idf}}
    '''

    tf_idf_out = {}

    for idx in tf_in.keys():

        tf_idf_out[idx] = {}

        tf_doc = tf_in[idx][0]
        len_doc = tf_in[idx][1]

        for word in tf_doc.keys():

            if d.check(word) and word in trained_words:
                tf_idf_out[idx][word] = (1.0 * tf_doc[word] / len_doc) * idf_in[word]

    return tf_idf_out


def sort_dict(dictionary, num_words):
    '''
    INPUT
    ----------
    dictionary : a dictionary variable
    top_N : how many entries you want to extract from the sorted dictionary (integer)

    OUTPUT
    ----------
    top_dict = [(key, value)] list of N tuples (key, value) wrt the value
                in descending order
    '''
    top_dict = [(k, dictionary[k]) for k in sorted(dictionary, key=dictionary.get,
                                                   reverse=True)][:num_words]

    return top_dict


def extract_top_words_docs(tf_idf_in, perc, num_words):
    '''
    INPUT
    ----------
    tf_idf : {doc_ID : {word : tf_idf}}
    perc: percentage of keywords to extract from the document
    max_N : how many entries (at most) you want to extract from the sorted dictionary (integer)

    OUTPUT
    ----------
    top_words : {word : [tf_idf scores]}} (if a word is in the Top_N words for more than
                                            a document, we'll have a number of Tf-idf values
                                            equal to the number of times this happens)
    '''

    top_words_doc = {}

    for idx in tf_idf_in.keys():

        len_reviews = len(tf_idf_in[idx])
        num_keywords = int(round(len_reviews * perc))

        if num_keywords > num_words:
            num_keywords = num_words

        if num_keywords == 0:
            num_keywords = 1

        top_words_doc[idx] = sort_dict(tf_idf_in[idx], num_keywords)

    return top_words_doc


def compute_coef(constant, n):
    '''
    INPUT
    ----------
    c : costant
    n : number of overlapping

    OUTPUT
    ----------
    coef : result of the computation
    '''

    coef = 0.

    for i in range(1, n + 1):
        coef += np.exp(- constant * (i - 1))

    return coef


def ranked_words(top_words_in, constant, words_dict):
    '''
    INPUT
    ----------
    topWords : {word :  [tf_idf scores]}
    c : costant
    d: PyEnchant dictionary containing meaningful English words

    OUTPUT
    ----------
    ranked_words : {word : score}
    '''

    ranked_words_out = {}

    for word in top_words_in.keys():

        if words_dict.check(word) and len(word) > 2:
            tf_idf_scores = top_words_in[word]
            n_overlap = len(tf_idf_scores)
            coef = compute_coef(constant, n_overlap)
            ranked_words_out[word] = max(tf_idf_scores) * coef

    return ranked_words_out


def keywords_dict(keywords_list, zero_based=False):
    keywords = {}

    if zero_based:
        idx = 0
        max_idx = 2000

    else:
        idx = 1
        max_idx = 2001

    for k, v in keywords_list:
        if keywords.get(k, None) is None and idx < max_idx:
            keywords[k] = idx
            idx += 1

    return keywords


def create_keywords_maps(keywords_dictionary, zero_based=False):
    keywords_map_out = {}
    keywords_inv_map_out = {}
    sorted_keywords = sorted(keywords_dictionary.keys())

    if zero_based:
        idx = 0

    else:
        idx = 1

    for keyword in sorted_keywords:
        keywords_map_out[keyword] = idx
        keywords_inv_map_out[idx] = keyword
        idx += 1

    return keywords_map_out, keywords_inv_map_out, sorted_keywords


def reviews_filtering(reviews_list, keywords_map_dict, max_len=20):
    output_list = []
    keys = set(keywords_map_dict.keys())
    kw_frequency = Counter()

    for review in reviews_list:
        user_ID = review[0]
        prod_ID = review[1]
        text = review[2]
        rating = review[3]
        keywords_review = []
        length = 0
        orig_len = len(text)

        for word in text:
            if word in keys and length < max_len:
                keywords_review.append(keywords_map_dict[word])
                length += 1

        final_len = len(keywords_review)

        if final_len > 1:
            kw_frequency += Counter(keywords_review)
            output_list.append([user_ID, prod_ID, keywords_review, rating, final_len, orig_len])

    print 'Original reviews:', len(reviews_list)
    print 'Final reviews:', len(output_list)
    print

    return output_list, kw_frequency


def create_keywords_matrix(sorted_keywords, model_w2v, zero_based=False, normalized=True):
    if zero_based:
        keywords_mat_out = []  # L2-normalized version

    else:
        keywords_mat_out = [np.zeros(utils.NUM_FEATURES)]  # L2-normalized version

    for word in sorted_keywords:
        keywords_mat_out.append(model_w2v.wv.word_vec(word, use_norm=normalized))

    print 'Number of keywords:', len(keywords_mat_out)
    print 'Vector dimensionality:', len(keywords_mat_out[0])

    return keywords_mat_out


# K-core and category selection
K = utils.K
name_core = str(K) + 'core'

categories = utils.CATEGORIES
for category in categories:

    print 'Processing category:', category
    DIR_PREPROCESSED_DATA_c = DIR_PREPROCESSED_DATA + '/' + category + '/' + name_core
    print DIR_PREPROCESSED_DATA_c
    DIR_TRAIN_TEST_DATA_c = DIR_TRAIN_TEST_DATA + '/' + category + '/' + name_core

    users_map = pickle.load(open(DIR_PREPROCESSED_DATA_c + '/users_map.pkl', 'rb'))
    products_map = pickle.load(open(DIR_PREPROCESSED_DATA_c + '/products_map.pkl', 'rb'))

    # ==============================================================================
    # KEYWORDS EXTRACTION
    # ==============================================================================
    top_N = utils.N_TOP_WORDS
    c = utils.C
    percentage = 0.1

    # load df and tf for all corpus
    print '\nLoading files..\n'
    tf = pickle.load(open(DIR_PREPROCESSED_DATA_c + '/tf_allcorpus.pkl', 'rb'))
    df = pickle.load(open(DIR_PREPROCESSED_DATA_c + '/df_allcorpus.pkl', 'rb'))

    print 'Computing TF-IDF..\n'
    N_docs = len(tf)
    idf = compute_idf(df, N_docs)
    tf_idf = compute_tf_idf(tf, idf)
    topN_words = extract_top_words_docs(tf_idf, perc=percentage, num_words=top_N)
    # ranking_dict = ranked_words(topN_words, c, d)

    print 'Extracting keywords...\n'
    sorted_list = sorted([item for l in topN_words.values() for item in l], \
                         key=lambda tup: tup[1], reverse=True)

    zero_based_flag = True
    kw_dict = keywords_dict(sorted_list, zero_based=zero_based_flag)
    keywords_map, keywords_inv_map, sorted_kw_list = \
    create_keywords_maps(kw_dict, zero_based=zero_based_flag)

    # ==============================================================================
    # REVIEWS FILTERING WRT KEYWORDS
    # ==============================================================================
    reviews = pickle.load(open(DIR_PREPROCESSED_DATA_c + '/reviews.pkl', 'rb'))
    output, keywords_freq = reviews_filtering(reviews, keywords_map)

    # ==============================================================================
    # VECTOR REPRESENTATION OF KEYWORDS
    # ==============================================================================

    print 'Saving output files..\n'

    keywords_mat = create_keywords_matrix(sorted_kw_list, model, zero_based=zero_based_flag)

    if not os.path.exists(DIR_TRAIN_TEST_DATA_c):
        print '\tCreate new output directory: ', DIR_TRAIN_TEST_DATA_c
        os.makedirs(DIR_TRAIN_TEST_DATA_c)

    with open(DIR_TRAIN_TEST_DATA_c + '/keywords_mat.pkl', "w") as outfile:
        pickle.dump(np.array(keywords_mat), outfile)
        outfile.close()

    with open(DIR_PREPROCESSED_DATA_c + '/keywords_map.pkl', "w") as outfile:
        pickle.dump(keywords_map, outfile)
        outfile.close()

    with open(DIR_PREPROCESSED_DATA_c + '/keywords_inv_map.pkl', "w") as outfile:
        pickle.dump(keywords_inv_map, outfile)
        outfile.close()

    with open(DIR_PREPROCESSED_DATA_c + '/reviews_keywordslist.pkl', "w") as outfile:
        pickle.dump(output, outfile)
        outfile.close()

    with open(DIR_PREPROCESSED_DATA_c + '/kw_frequency.pkl', "w") as outfile:
        pickle.dump(keywords_freq, outfile)
        outfile.close()

    print
    print
