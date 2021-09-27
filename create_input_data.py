#        Interpreting Node Embedding with Text-Labeled Graphs
	  
#   File:     create_input_data.py 
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
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec

import utils
from utils import count_reviews, mapping

MODEL_PATH = utils.MODEL_PATH
DIR_PREPROCESSED_DATA = utils.DIR_PREPROCESSED_DATA
DIR_TRAIN_TEST_DATA = utils.DIR_TRAIN_TEST_DATA

print 'Loading data.. \n'
# load the model
word_vectors = Word2Vec.load(MODEL_PATH)
word_vectors.init_sims()  # to get L2-normalized word vectors


def save_embedding_matrix(w_vectors, inverse_map):
    M = [np.zeros(utils.NUM_FEATURES)]

    for i in range(1, len(inverse_map.keys()) + 1):
        M.append(w_vectors[inverse_map[i]])

    print 'Saving embedding matrix..\n'
    pickle.dump(np.array(M), open(DIR_TRAIN_TEST_DATA + '/embedding_matrix.pkl', 'w'))


def create_training_data(reviews, dir_output, users_mapping, products_mapping, users_dict, max_len=20):
    random.shuffle(reviews)     # in-place shuffling
    user_IDs = []
    product_IDs = []
    reviews_out = []
    ratings = []

    for line in tqdm(reviews):
        user_ID = line[0]
        product_ID = line[1]

        if user_ID in users_set and product_ID in products_set:
            idx_u = users_mapping[user_ID]
            idx_p = products_mapping[product_ID]
            review = line[2][:max_len]
            rating = line[3]
            norm_rating = rating - users_dict[user_ID]['mean']
            user_IDs.append(idx_u)
            product_IDs.append(idx_p)
            reviews_out.append(review)
            ratings.append(norm_rating)

    # shuffling
    seed = random.randint(0, 10e6)
    random.seed(seed)
    random.shuffle(user_IDs)
    random.seed(seed)
    random.shuffle(product_IDs)
    random.seed(seed)
    random.shuffle(reviews_out)
    random.seed(seed)
    random.shuffle(ratings)

    print len(user_IDs), len(set(user_IDs))
    print len(product_IDs), len(set(product_IDs))
    print len(reviews_out), len(ratings)
    print
    print 'Saving files..'

    with open(dir_output + '/user_IDs.pkl', "w") as outfile:
        pickle.dump(user_IDs, outfile)
        outfile.close()

    with open(dir_output + '/product_IDs.pkl', "w") as outfile:
        pickle.dump(product_IDs, outfile)
        outfile.close()

    with open(dir_output + '/ratings.pkl', "w") as outfile:
        pickle.dump(ratings, outfile)
        outfile.close()

    with open(dir_output + '/reviews_out.pkl', "w") as outfile:
        pickle.dump(reviews_out, outfile)
        outfile.close()

    return len(set(user_IDs)), len(set(product_IDs)), len(reviews)


# K-core and category selection
K = utils.K
name_core = str(K) + 'core'
categories = utils.CATEGORIES
n_users = []
n_prods = []
n_reviews = []
for category in categories:
    print 'Processing category:', category
    DIR_PREPROCESSED_DATA_c = DIR_PREPROCESSED_DATA + '/' + category + '/' + name_core
    DIR_TRAIN_TEST_DATA_c = DIR_TRAIN_TEST_DATA + '/' + category + '/' + name_core
    
    # load data (reviewerID i , asin j , reviewText_ij, rating ij)
    indexed_kw_reviews = pickle.load(
        open(DIR_PREPROCESSED_DATA_c + '/reviews_keywordslist.pkl', 'rb')) 

    users, products = count_reviews(indexed_kw_reviews, filter_flag=True)
    users_set = users.keys()
    products_set = products.keys()
    print len(users_set), len(products_set)
    users_map, products_map = mapping(users_set, products_set)
    print 'Creating training data..\n'
    n_u, n_p, n_r = create_training_data(indexed_kw_reviews, DIR_PREPROCESSED_DATA_c, users_map, products_map, users)

    n_users.append(n_u)
    n_prods.append(n_p)
    n_reviews.append(n_r)
    pickle.dump(users_map, open(DIR_TRAIN_TEST_DATA_c + '/users_map.pkl', "w"))
    pickle.dump(products_map, open(DIR_TRAIN_TEST_DATA_c + '/products_map.pkl', "w"))
    print
    print

tem = {'category': categories, 'n_users': n_users, 'n_prods': n_prods, 'n_reviews': n_reviews}
df = pd.DataFrame(tem)
df.to_csv(DIR_PREPROCESSED_DATA + '/stat_after.csv', index=False)