#        Interpreting Node Embedding with Text-Labeled Graphs
	  
#   File:     data_split.py 
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
import itertools
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from sklearn.model_selection import train_test_split
import utils

DIR_PREPROCESSED_DATA = utils.DIR_PREPROCESSED_DATA
DIR_TRAIN_TEST_DATA = utils.DIR_TRAIN_TEST_DATA

# K-core and category selection
K = utils.K
name_core = str(K) + 'core'

categories = utils.CATEGORIES
for category in categories:
    print 'Processing category:', category

    DIR_PREPROCESSED_DATA_c = DIR_PREPROCESSED_DATA + '/' + category + '/' + name_core
    DIR_TRAIN_TEST_DATA_c = DIR_TRAIN_TEST_DATA + '/' + category + '/' + name_core

    # load data
    print 'Loading data.. \n'
    user_IDs = pickle.load(open(DIR_PREPROCESSED_DATA_c + '/user_IDs.pkl', 'rb'))
    product_IDs = pickle.load(open(DIR_PREPROCESSED_DATA_c + '/product_IDs.pkl', 'rb'))
    reviews = pickle.load(open(DIR_PREPROCESSED_DATA_c + '/reviews_out.pkl', 'rb'))
    ratings = pickle.load(open(DIR_PREPROCESSED_DATA_c + '/ratings.pkl', 'rb'))

    # training test splitting
    idx_tr, idx_tev = train_test_split(range(len(reviews)), test_size=.2, shuffle=False)
    idx_te, idx_val = train_test_split(idx_tev, test_size=.5, shuffle=False)

    users_ID_tr = np.array([user_IDs[i] for i in idx_tr])
    users_ID_te_tem = np.array([user_IDs[i] for i in idx_te])
    prods_ID_tr = np.array([product_IDs[i] for i in idx_tr])
    prods_ID_te_tem = np.array([product_IDs[i] for i in idx_te])
    reviews_tr = np.array([reviews[i] for i in idx_tr])
    reviews_te_tem = np.array([reviews[i] for i in idx_te])
    ratings_tr = np.array([ratings[i] for i in idx_tr])
    ratings_te_tem = np.array([ratings[i] for i in idx_te])

    set_users_train = set(users_ID_tr)
    set_products_train = set(prods_ID_tr)

    reviews_te = []
    ratings_te = []
    prods_ID_te = []
    users_ID_te = []
    print len(reviews_te_tem), len(ratings_te_tem), len(users_ID_te_tem), len(prods_ID_te_tem)
    
    for i in range(len(reviews_te_tem)):
        user_id = users_ID_te_tem[i]
        prod_id = prods_ID_te_tem[i]
        review = reviews_te_tem[i]
        rating = ratings_te_tem[i]

        if user_id in set_users_train and prod_id in set_products_train:
            reviews_te.append(review)
            users_ID_te.append(user_id)
            prods_ID_te.append(prod_id)
            ratings_te.append(rating)

    reviews_te = np.array(reviews_te)
    ratings_te = np.array(ratings_te)
    users_ID_te = np.array(users_ID_te)
    prods_ID_te = np.array(prods_ID_te)

    print len(reviews_te), len(ratings_te), len(users_ID_te), len(prods_ID_te)
    users_id_train = []
    products_id_train = []
    words_train = []
    ratings_train = []

    for i in range(len(reviews_tr)):
        user_id = users_ID_tr[i]
        prod_id = prods_ID_tr[i]
        review = reviews_tr[i]
        rating = ratings_tr[i]
        combinations = list(itertools.combinations(review, 2))

        for combination in combinations:
            words_train.append(combination)
            users_id_train.append(user_id)
            products_id_train.append(prod_id)
            ratings_train.append(rating)

    users_id_train = np.array(users_id_train)
    products_id_train = np.array(products_id_train)
    words_train = np.array(words_train)
    ratings_train = np.array(ratings_train)

    users_id_test = []
    products_id_test = []
    words_test = []
    ratings_test = []

    for i in range(len(reviews_te)):
        user_id = users_ID_te[i]
        prod_id = prods_ID_te[i]
        review = reviews_te[i]
        rating = ratings_te[i]
        combinations = list(itertools.combinations(review, 2))

        for combination in combinations:
            words_test.append(combination)
            users_id_test.append(user_id)
            products_id_test.append(prod_id)
            ratings_test.append(rating)

    users_id_test = np.array(users_id_test)
    products_id_test = np.array(products_id_test)
    words_test = np.array(words_test)
    ratings_test = np.array(ratings_test)

    print users_id_train.shape
    print users_id_test.shape
    print products_id_train.shape
    print products_id_test.shape
    print words_train.shape
    print words_test.shape
    print ratings_train.shape
    print ratings_test.shape
    print

    if not os.path.exists(DIR_TRAIN_TEST_DATA_c):
        print '\tCreate new output directory: ', DIR_TRAIN_TEST_DATA_c
        os.makedirs(DIR_TRAIN_TEST_DATA_c)

    print
    print 'Saving files..'
    pickle.dump(users_id_train, open(DIR_TRAIN_TEST_DATA_c + '/users_ID_train.pkl', "w"))
    pickle.dump(users_id_test, open(DIR_TRAIN_TEST_DATA_c + '/users_ID_test.pkl', "w"))
    pickle.dump(products_id_train, open(DIR_TRAIN_TEST_DATA_c + '/prods_ID_train.pkl', "w"))
    pickle.dump(products_id_test, open(DIR_TRAIN_TEST_DATA_c + '/prods_ID_test.pkl', "w"))
    pickle.dump(words_train, open(DIR_TRAIN_TEST_DATA_c + '/words_train.pkl', "w"))
    pickle.dump(words_test, open(DIR_TRAIN_TEST_DATA_c + '/words_test.pkl', "w"))
    pickle.dump(ratings_train, open(DIR_TRAIN_TEST_DATA_c + '/ratings_train.pkl', "w"))
    pickle.dump(ratings_test, open(DIR_TRAIN_TEST_DATA_c + '/ratings_test.pkl', "w"))

    print 'Done!'
    print
    print
