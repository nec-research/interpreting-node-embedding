#        Interpreting Node Embedding with Text-Labeled Graphs
	  
#   File:     preprocessing_raw_text.py 
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
import gzip
import nltk
import pickle
import enchant
import pandas as pd
import networkx as nx
from collections import Counter
from gensim.models import Word2Vec
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import utils
from utils import mapping

d = enchant.Dict('EN')
wnl = WordNetLemmatizer()

DIR_DATA = utils.DIR_DATA
DIR_PREPROCESSED_DATA = utils.DIR_PREPROCESSED_DATA
MODEL_PATH = utils.MODEL_PATH

stopwords = utils.STOPWORDS

# load the model
word_vectors = Word2Vec.load(MODEL_PATH)
rank = word_vectors.wv.index2word
words_set = set(rank)
window_size = 1

SENTENCE_SPLIT_REGEX = utils.SENTENCE_SPLIT_REGEX
D = utils.NUM_FEATURES

def parse(dir_folder):
    g = gzip.open(dir_folder, 'rb')
    for l in g:
        yield eval(l)


def split_sentence(sentence, stoplist=stopwords, trained_words=words_set):
    sentence = [s.lower().strip() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) \
                if len(s.strip()) > 2 and s.lower().strip() not in stoplist \
                and s.lower().strip() in trained_words and d.check(s.lower().strip())]
    return sentence


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def reviews_filter(dictionary, threshold):
    print
    print 'Initial number of keys: {0}'.format(len(dictionary.keys()))

    output = dict((k, v) for k, v in dictionary.iteritems() if v >= threshold)
    print 'Number of keys after filtering: {0}'.format(len(output.keys()))

    perc = float(len(output.keys())) / len(dictionary.keys()) * 100
    print 'Percentage of keys after filtering: {0:.2f}%\n'.format(perc)

    return set(output.keys())


def create_graph(dir_data):
    count = 0
    G = nx.Graph()
    dict_users = {}
    dict_products = {}

    for filename in os.listdir(dir_data):
        print
        print 'Processing folder: {0}\n'.format(filename)

        for review in parse(os.path.join(dir_data, filename)):
            asin = review['asin']
            reviewerID = review['reviewerID']
            G.add_edge(reviewerID, asin)

            if dict_users.get(reviewerID, None) is None:
                dict_users[reviewerID] = 1
            else:
                dict_users[reviewerID] += 1

            if dict_products.get(asin, None) is None:
                dict_products[asin] = 1
            else:
                dict_products[asin] += 1

            count += 1
            if count % 100000 == 0:
                print 'Done {0} sentences'.format(count)

    users = set(dict_users.keys())
    products = set(dict_products.keys())
    return G, users, products


def compute_Kcore(G, k, users, products):
    K_G = nx.k_core(G, k)
    nodes = set(K_G.nodes())
    set_users = users.intersection(nodes)
    set_products = products.intersection(nodes)

    print
    print 'Initial number of users: {0}'.format(len(users))
    print 'Number of users after filtering: {0}'.format(len(set_users))
    print

    print 'Initial number of products: {0}'.format(len(products))
    print 'Number of products after filtering: {0}'.format(len(set_products))
    print

    perc_u = float(len(set_users)) / len(users) * 100
    print 'Percentage of users after filtering: {0:.2f}%'.format(perc_u)

    perc_p = float(len(set_products)) / len(products) * 100
    print 'Percentage of products after filtering: {0:.2f}%\n'.format(perc_p)

    return set_users, set_products


def preprocess_data(dir_data, dir_output, users_set, products_set):
    df = Counter()
    tf = {}
    idx = 0
    reviews_out = []

    for foldername in os.listdir(dir_data):
        subfolder = os.path.join(dir_data, foldername)

        if foldername == 'reviews':
            for filename in os.listdir(subfolder):
                print
                print 'Processing folder: {0}\n'.format(filename)

                for line in gzip.open(os.path.join(subfolder, filename)):
                    data = eval(line)
                    # maximum length of raw documents = 300
                    reviewText = data['reviewText'][:300]  
                    asin = data['asin']
                    reviewerID = data['reviewerID']
                    rating = data['overall']

                    if reviewerID in users_set and asin in products_set:
                        reviewText_split = split_sentence(reviewText.decode('utf-8'))
                        reviewText_processed = [wnl.lemmatize(w, get_wordnet_pos(w)) for w in reviewText_split]

                        if len(reviewText_processed) > window_size and len(set(reviewText_processed)) > 1:
                            tf[idx] = [Counter(reviewText_processed), len(reviewText_processed)]
                            reviews_out.append([reviewerID, asin, reviewText_processed, rating])

                            for word in set(reviewText_processed):
                                df[word] += 1

                            idx += 1

    print 'Saving files..'

    with open(dir_output + '/reviews.pkl', "w") as outfile:
        pickle.dump(reviews_out, outfile)
        outfile.close()

    with open(dir_output + '/df_allcorpus.pkl', "w") as outfile:
        pickle.dump(df, outfile)
        outfile.close()

    with open(dir_output + '/tf_allcorpus.pkl', "w") as outfile:
        pickle.dump(tf, outfile)
        outfile.close()

    return reviews_out


def create_maps(rank_list, dir_output):
    words_map = {}
    inv_words_map = {}

    for idx, word in enumerate(rank_list):
        words_map[word] = idx + 1
        inv_words_map[idx + 1] = word

    print 'Saving maps.. \n'
    with open(dir_output + '/words_map.pkl', "w") as outfile:
        pickle.dump(words_map, outfile)
        outfile.close()

    with open(dir_output + '/inv_words_map.pkl', "w") as outfile:
        pickle.dump(inv_words_map, outfile)
        outfile.close()

    return words_map, inv_words_map


def create_indexed_reviews(review_list, words_mapping, dir_output):
    output = []
    for review in review_list:
        userID = review[0]
        itemID = review[1]
        reviewText = review[2]
        rating = review[3]
        indexed_review = []

        for word in reviewText:
            indexed_review.append(words_mapping[word])

        output.append([userID, itemID, indexed_review, rating])

    print 'Saving indexed reviews.. \n'
    with open(dir_output + '/indexed_reviews.pkl', "w") as outfile:
        pickle.dump(output, outfile)
        outfile.close()


# K-core and category selection
K = utils.K
name_core = str(K) + 'core'

categories = utils.CATEGORIES
# categories = ['baby']

n_users = []
n_prods = []
n_reviews = []

for category in categories:
    print 'Processing category:', category
    print
    DIR_DATA_c = DIR_DATA + '/' + category
    DIR_REVIEWS_c = DIR_DATA_c + '/reviews/'
    DIR_PREPROCESSED_DATA_c = DIR_PREPROCESSED_DATA + '/' + category + '/' + name_core

    # remove users and products with less than K reviews each
    G, all_users, all_products = create_graph(DIR_REVIEWS_c)

    # extract K-core
    Kcore_users, Kcore_products = compute_Kcore(G, K, all_users, all_products)

    if not os.path.exists(DIR_PREPROCESSED_DATA_c):
        print '\tCreate new output directory: ', DIR_PREPROCESSED_DATA_c
        os.makedirs(DIR_PREPROCESSED_DATA_c)

    # preprocess data
    users_map, products_map = mapping(Kcore_users, Kcore_products)
    print 'Number of users:', len(users_map)
    print 'Number of products:', len(products_map)

    n_users.append(len(users_map))
    n_prods.append(len(products_map))
    pickle.dump(users_map, open(DIR_PREPROCESSED_DATA_c + '/users_map.pkl', "w"))
    pickle.dump(products_map, open(DIR_PREPROCESSED_DATA_c + '/products_map.pkl', "w"))

    reviews = preprocess_data(DIR_DATA_c, DIR_PREPROCESSED_DATA_c, Kcore_users, Kcore_products)
    print 'Number of reviews:', len(reviews)
    n_reviews.append(len(reviews))

    # create indexed reviews
    words_map, inv_words_map = create_maps(rank, DIR_PREPROCESSED_DATA_c)
    # create_indexed_reviews(reviews, words_map)
    print
    print
    print

tem = {'category': categories, 'n_users': n_users, 'n_prods': n_prods, 'n_reviews': n_reviews}
df = pd.DataFrame(tem)
df.to_csv(DIR_PREPROCESSED_DATA + '/stat_original.csv', index=False)
