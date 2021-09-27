#        Interpreting Node Embedding with Text-Labeled Graphs
      
#   File:     run_ignn.py 
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
import keras
import time
import pickle
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

# GPU options
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Hint: so the IDs match nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

flatten = keras.layers.Flatten  # Flatten = tf.layers.flatten (deprecated)
dropout = keras.layers.Dropout  # tf.layers.dropout/tf.nn.dropout (deprecated)
dense = keras.layers.Dense  # tf.layers.dense/tf.nn.dense (deprecated)

DIR_TRAIN_TEST_DATA = utils.DIR_TRAIN_TEST_DATA
DIR_RESULTS = utils.DIR_RESULTS

# K-core and category selection
K = utils.K
name_core = str(K) + 'core'
categories = utils.CATEGORIES
name_arch = 'iGNN'

for category in categories:
    print 'Processing category:', category
    DIR_TRAIN_TEST_DATA_c = DIR_TRAIN_TEST_DATA + '/' + category + '/' + name_core
    DIR_RESULTS_c = DIR_RESULTS + '/' + category + '/' + name_core + '/' + name_arch

    if not os.path.exists(DIR_RESULTS_c):
        print '\tCreate new output directory: ', DIR_RESULTS_c
        os.makedirs(DIR_RESULTS_c)

    # load training data
    users_map = pickle.load(open(DIR_TRAIN_TEST_DATA_c + '/users_map.pkl', 'rb'))
    products_map = pickle.load(open(DIR_TRAIN_TEST_DATA_c + '/products_map.pkl', 'rb'))
    users_ID_train = pickle.load(open(DIR_TRAIN_TEST_DATA_c + '/users_ID_train.pkl', 'rb'))
    users_ID_test = pickle.load(open(DIR_TRAIN_TEST_DATA_c + '/users_ID_test.pkl', 'rb'))
    prods_ID_train = pickle.load(open(DIR_TRAIN_TEST_DATA_c + '/prods_ID_train.pkl', 'rb'))
    prods_ID_test = pickle.load(open(DIR_TRAIN_TEST_DATA_c + '/prods_ID_test.pkl', 'rb'))
    words_train = pickle.load(open(DIR_TRAIN_TEST_DATA_c + '/words_train.pkl', 'rb'))
    words_test = pickle.load(open(DIR_TRAIN_TEST_DATA_c + '/words_test.pkl', 'rb'))
    ratings_train = pickle.load(open(DIR_TRAIN_TEST_DATA_c + '/ratings_train.pkl', 'rb'))
    ratings_test = pickle.load(open(DIR_TRAIN_TEST_DATA_c + '/ratings_test.pkl', 'rb'))

    print len(ratings_train), len(users_ID_train), len(prods_ID_train)

    # keyword embedding parameters
    wordEmb = pickle.load(open(DIR_TRAIN_TEST_DATA_c + "/keywords_mat.pkl", 'rb')) #(VxD)
    size_voc = wordEmb.shape[0]
    dim_words = wordEmb.shape[1]
    print wordEmb.shape

    num_users = len(users_map.keys())
    num_prods = len(products_map.keys())
    print num_users, num_prods

    # embedding hyperparameters
    dim_emb_users = 200
    dim_emb_prods = 200
    num_cluster_users = 50
    num_cluster_prods = 30
    num_cluster_combinations = num_cluster_users * num_cluster_prods
    sum_l = [np.arange(num_cluster_prods) + num_cluster_prods * i for i in range(num_cluster_users)]
    sum_k = [np.arange(0, num_cluster_users * num_cluster_prods, num_cluster_prods) + 1 * i for i in
             range(num_cluster_prods)]

    dim_cluster_users = 200
    dim_cluster_prods = 200
    num_combinations = words_train.shape[1]
    tau0 = 0.5  # or 1.0
    isTempLearnable = False
    batch_size = 512
    num_epochs = 201
    threshold = 1. / (num_cluster_prods * num_cluster_users)
    lambda_sparsegen_u = 0.95  # term for sparsegen
    # lambda_sparsegen_u = 0.  # term for sparsegen
    coef_sparsegen_u = 1. / (1 - lambda_sparsegen_u)
    lambda_sparsegen_p = 0.75  # term for sparsegen
    # lambda_sparsegen_p = 0.  # term for sparsegen
    coef_sparsegen_p = 1. / (1 - lambda_sparsegen_p)

    # optimization hyperparameters
    lr = 2e-6
    loss_weights = [.2, 1.]

    # MLP hyperparameters
    num_units_net = [128, 64, 64, 32]
    dropout_rate_net = [.5, .5, .5, .3]
    c_ratings = np.ceil(np.max(np.abs(ratings_train)))

    # TF graph initialization
    tf.reset_default_graph()
    users_ID = tf.placeholder(tf.int32, shape=(None,))
    prods_ID = tf.placeholder(tf.int32, shape=(None,))
    words = tf.placeholder(tf.int32, shape=(None, 2))
    ratings = tf.placeholder(tf.float32, shape=(None,))
    ratings_expanded = tf.expand_dims(ratings, 1)  # needed for computing loss
    training_phase = tf.placeholder_with_default(True, shape=())

    print '\tembedding of words:'
    wordEmb_init = tf.Variable(tf.constant(0.0, shape=[size_voc, dim_words]),
                               trainable=False, name='embedding_words')
    wordEmb_placeholder = tf.placeholder(tf.float32, [size_voc, dim_words])
    embedding_words = wordEmb_init.assign(wordEmb_placeholder)
    print '\tembedding_words =', embedding_words
    print

    print '\tembedding of users:'
    embedding_users = tf.get_variable('embedding_users', [num_users, dim_emb_users])
    x_users_lu = tf.nn.embedding_lookup(embedding_users, users_ID)
    x_users = dropout(rate=.4)(x_users_lu, training=training_phase)
    print '\tembedding_users =', embedding_users
    print '\tx_users =', x_users
    print

    print '\tembedding of products:'
    embedding_prods = tf.get_variable('embedding_prods', [num_prods, dim_emb_prods])
    x_prods_lu = tf.nn.embedding_lookup(embedding_prods, prods_ID)
    x_prods = dropout(rate=.4)(x_prods_lu, training=training_phase)
    print '\tembedding_prods =', embedding_prods
    print '\tx_prods =', x_prods
    print

    print '\tembedding of user clusters:'
    embedding_clusters_user = tf.get_variable('embedding_clusters_user',
                                              [num_cluster_users, dim_cluster_users])
    print '\tembedding_clusters_user =', embedding_clusters_user
    print

    print '\tembedding of prod clusters:'
    embedding_clusters_prod = tf.get_variable('embedding_clusters_prod',
                                              [num_cluster_prods, dim_cluster_prods])
    print '\tembedding_clusters_prod =', embedding_clusters_prod
    print

    print '\tTheta of user'
    with tf.variable_scope('global_vars'):
        W_theta_user = tf.get_variable('W_theta_user', [dim_emb_users, dim_cluster_users])
        tau = tf.Variable(tau0, name='temperature', trainable=isTempLearnable)

    tem_theta_user = tf.matmul(x_users, W_theta_user)
    logits_theta_user_tem = tf.matmul(tem_theta_user, tf.transpose(embedding_clusters_user))
    logits_theta_user = tf.multiply(logits_theta_user_tem, coef_sparsegen_u)
    print '\tlogits_theta_user =', logits_theta_user

    z_user = tf.contrib.sparsemax.sparsemax(logits_theta_user)
    print '\tz_user =', z_user
    print

    print '\tTheta of prod'
    with tf.variable_scope('global_vars'):
        W_theta_prod = tf.get_variable('W_theta_prod', [dim_emb_prods, dim_cluster_prods])

    tem_theta_prod = tf.matmul(x_prods, W_theta_prod)
    logits_theta_prod_tem = tf.matmul(tem_theta_prod, tf.transpose(embedding_clusters_prod))
    logits_theta_prod = tf.multiply(logits_theta_prod_tem, coef_sparsegen_p)
    print '\tlogits_theta_prod =', logits_theta_prod

    z_prod = tf.contrib.sparsemax.sparsemax(logits_theta_prod)
    print '\tz_prod =', z_prod
    print

    print '\tBeta'
    with tf.variable_scope('global_vars'):
        W_beta = tf.get_variable('W_beta', [(dim_cluster_prods + dim_cluster_users), dim_words])

    tem_beta = tf.matmul(W_beta, tf.transpose(embedding_words))
    print '\ttem_beta', tem_beta

    k_users = np.repeat(np.array(range(num_cluster_users)), num_cluster_prods)
    k_prods = np.tile(np.array(range(num_cluster_prods)), num_cluster_users)

    cluster_users = tf.nn.embedding_lookup(embedding_clusters_user, k_users)
    cluster_users = dropout(rate=.3)(cluster_users, training=training_phase)
    print '\tcluster_users', cluster_users

    cluster_prods = tf.nn.embedding_lookup(embedding_clusters_prod, k_prods)
    cluster_prods = dropout(rate=.3)(cluster_prods, training=training_phase)
    print '\tcluster_prods', cluster_prods

    concat_clusters = tf.concat([cluster_users, cluster_prods], axis=1)
    print '\tconcat_clusters', concat_clusters

    logits_beta = tf.matmul(concat_clusters, tem_beta)
    beta = tf.contrib.sparsemax.sparsemax(logits_beta)
    # beta = tf.nn.softmax(logits_beta)
    print '\tbeta', beta

    print
    print '\tlikelihood of words:'

    tem_Z = tf.matmul(tf.expand_dims(z_user, axis=-1), tf.expand_dims(z_prod, axis=1))
    print '\ttf.expand_dims(z_user, axis = -1)', tf.expand_dims(z_user, axis=-1)
    print '\ttf.expand_dims(z_prod, axis = 1)', tf.expand_dims(z_prod, axis=1)
    print '\ttem_Z', tem_Z

    Z_single = flatten()(tem_Z)
    print '\tZ_single', Z_single

    # we have a bigram model (w1, w2) ---> we repeat Z_single twice
    Z = tf.reshape(tf.tile(Z_single, [1, 2]), [tf.shape(Z_single)[0], 2, num_cluster_combinations])
    print '\tZ', Z

    # ==============================================================================
    #     RATING PREDICTION PART
    # ==============================================================================
    embedding_ij = tf.concat([x_users, x_prods], axis=-1)
    print '\tembedding_ij', embedding_ij
    print

    net_ratings_1 = dense(units=num_units_net[0], activation=tf.nn.relu)(embedding_ij)
    drop_ratings_1 = dropout(rate=dropout_rate_net[0])(net_ratings_1, training=training_phase)
    net_ratings_2 = dense(units=num_units_net[1], activation=tf.nn.relu)(drop_ratings_1)
    drop_ratings_2 = dropout(rate=dropout_rate_net[1])(net_ratings_2, training=training_phase)
    net_ratings_3 = dense(units=num_units_net[2], activation=tf.nn.relu)(drop_ratings_2)
    drop_ratings_3 = dropout(rate=dropout_rate_net[2])(net_ratings_1, training=training_phase)
    net_ratings_4 = dense(units=num_units_net[3], activation=tf.nn.relu)(drop_ratings_3)
    drop_ratings_4 = dropout(rate=dropout_rate_net[3])(net_ratings_2, training=training_phase)

    tem_ratings = dense(units=1, activation=tf.nn.tanh)(drop_ratings_4)  # SCALED-TANH
    output_ratings = tem_ratings * c_ratings
    print '\toutput_ratings=', output_ratings

    # ==============================================================================
    #     LIKELIHOOD
    # ==============================================================================
    tem = tf.nn.embedding_lookup(tf.transpose(beta), words)
    likelihood_tem = tf.reduce_sum(tf.multiply(Z, tem), -1)
    likelihood = tf.reduce_mean(likelihood_tem, -1) + 1e-6
    neg_log_likelihood = -tf.reduce_mean(tf.math.log(likelihood), -1)

    print
    print '\ttem =', tem
    print '\tlikelihood_tem=', likelihood_tem
    print '\tlikelihood=', likelihood
    print '\tneg_log_likelihood =', neg_log_likelihood
    print

    # ==============================================================================
    #     LOSS and OPTIMIZER
    # ==============================================================================
    rating_loss = tf.losses.mean_squared_error(labels=ratings_expanded, \
                                               predictions=output_ratings)
    print '\tmse_loss =', rating_loss

    loss = (loss_weights[0]*neg_log_likelihood) + (loss_weights[1]*rating_loss)
    train_optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)

    # In[22]:
    # ==============================================================================
    #     DATASET ITERATOR
    # ==============================================================================
    train = tf.data.Dataset.from_tensor_slices((users_ID, prods_ID, words, ratings))
    train = train.shuffle(buffer_size=words_train.shape[0], \
                          reshuffle_each_iteration=True).batch(batch_size)
    test = tf.data.Dataset.from_tensor_slices((users_ID, prods_ID, words, ratings))
    test = test.batch(batch_size)

    iterator_train = train.make_initializable_iterator()
    iterator_test = test.make_initializable_iterator()
    next_batch_train = iterator_train.get_next()
    next_batch_test = iterator_test.get_next()
    show_step = 2

    try:
        with tf.train.MonitoredSession() as sess:
            log_nll_train = []
            log_nll_test = []
            log_mse_train = []
            log_mse_test = []

            for epoch in range(1, num_epochs):
                start_time = time.clock()
                dictplaceholders_train = {users_ID: users_ID_train,
                                          prods_ID: prods_ID_train,
                                          words: words_train,
                                          ratings: ratings_train,
                                          wordEmb_placeholder: wordEmb}
                sess.run(iterator_train.initializer, dictplaceholders_train)
                batch_nll_train = []
                batch_mse_train = []
                
                while True:
                    try:
                        users_tr, prods_tr, words_tr, ratings_tr = sess.run(next_batch_train)
                        res_batch = sess.run([train_optimizer, neg_log_likelihood, rating_loss, loss],
                                             {users_ID: users_tr, prods_ID: prods_tr,
                                              wordEmb_placeholder: wordEmb,
                                              words: words_tr, ratings: ratings_tr})

                        batch_nll_train.append(res_batch[1])
                        batch_mse_train.append(res_batch[2])

                    except tf.errors.OutOfRangeError:
                        break

                if epoch % show_step == 0:
                    dictplaceholders_test = {users_ID: users_ID_test,
                                             prods_ID: prods_ID_test,
                                             words: words_test,
                                             ratings: ratings_test,
                                             wordEmb_placeholder: wordEmb,
                                             training_phase: False}

                    sess.run(iterator_test.initializer, dictplaceholders_test)
                    batch_nll_test = []
                    batch_mse_test = []
                    
                    while True:
                        try:
                            users_te, prods_te, words_te, ratings_te = sess.run(next_batch_test)
                            res_batch_test = sess.run([neg_log_likelihood, rating_loss, loss],
                                                      {users_ID: users_te, prods_ID: prods_te,
                                                       wordEmb_placeholder: wordEmb,
                                                       words: words_te,
                                                       ratings: ratings_te,
                                                       training_phase: False})
                            batch_nll_test.append(res_batch_test[0])
                            batch_mse_test.append(res_batch_test[1])

                        except tf.errors.OutOfRangeError:
                            break

                    epoch_nll = np.mean(batch_nll_train)
                    test_nll = np.mean(batch_nll_test)
                    epoch_mse = np.mean(batch_mse_train)
                    test_mse = np.mean(batch_mse_test)
                    
                    log_nll_train.append(epoch_nll)
                    log_nll_test.append(test_nll)
                    log_mse_train.append(epoch_mse)
                    log_mse_test.append(test_mse)
                    
                    print(category + ' - Step %d, NLL__train: %0.3f, NLL__test: %0.3f'\
                          % (epoch, epoch_nll, test_nll))
                    print(category + ' - Step %d, MSE__train: %0.3f, MSE__test: %0.3f'\
                          % (epoch, epoch_mse, test_mse))
                    print                    
                    
                time_elapsed = (time.clock() - start_time)
                print time_elapsed
                print

            z_u, z_p, beta_out = sess.run([z_user, z_prod, beta],
                                          feed_dict={users_ID: users_map.values(),
                                                     prods_ID: products_map.values(),
                                                     wordEmb_placeholder: wordEmb,
                                                     training_phase: False})

    except KeyboardInterrupt:
        print '\tTraining interrupted by user. Saving outputs...'
        break

    with open(DIR_RESULTS_c + '/beta.pkl', 'w') as outfile:
        pickle.dump(beta_out, outfile)
        outfile.close()

    with open(DIR_RESULTS_c + '/z_users.pkl', 'w') as outfile:
        pickle.dump(z_u, outfile)
        outfile.close()

    with open(DIR_RESULTS_c + '/z_prods.pkl', 'w') as outfile:
        pickle.dump(z_p, outfile)
        outfile.close()

    tem = {'train_nll_loss': log_nll_train, 'test_nll_loss': log_nll_test}
    df = pd.DataFrame(tem)
    df.to_csv(DIR_RESULTS_c + '/nll_evaluation.csv', index=False)
    ticks = np.arange(0, len(df) + 1, 10)

    plt.plot(df['train_nll_loss'], label='Train')
    plt.plot(df['test_nll_loss'], label='Test')
    plt.legend()
    plt.title('NegLogLik Loss')
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.xticks(ticks, [i * show_step for i in ticks], rotation=70)
    plt.savefig(DIR_RESULTS_c + '/NLLcurve_original.pdf', format='pdf', \
                transparent=True, dpi=200, bbox_inches='tight')
    plt.savefig(DIR_RESULTS_c + '/NLLcurve_original.png', format='png', \
                transparent=True, dpi=200, bbox_inches='tight')
    plt.close()

    plt.plot(df['train_nll_loss'] * loss_weights[0], label='Train')
    plt.plot(df['test_nll_loss'] * loss_weights[0], label='Test')
    plt.legend()
    plt.title('NegLogLik Loss')
    plt.xlabel('Epoch')
    plt.ylabel('LogLik')
    plt.xticks(ticks, [i * show_step for i in ticks], rotation=70)
    plt.savefig(DIR_RESULTS_c + '/NLLcurve.pdf', format='pdf', \
                transparent=True, dpi=200, bbox_inches='tight')
    plt.savefig(DIR_RESULTS_c + '/NLLcurve.png', format='png', \
                transparent=True, dpi=200, bbox_inches='tight')
    plt.close()

    tem = {'train_mse_loss': log_mse_train, 'test_mse_loss': log_mse_test}
    df = pd.DataFrame(tem)
    df.to_csv(DIR_RESULTS_c + '/mse_evaluation.csv', index=False)

    plt.plot(df['train_mse_loss'], label='Train')
    plt.plot(df['test_mse_loss'], label='Test')
    plt.legend()
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.xticks(ticks, [i * show_step for i in ticks], rotation=70)
    plt.savefig(DIR_RESULTS_c + '/MSEcurve_original.pdf', format='pdf',\
                transparent=True, dpi=200, bbox_inches='tight')
    plt.savefig(DIR_RESULTS_c + '/MSEcurve_original.png', format='png',\
                transparent=True, dpi=200, bbox_inches='tight')
    plt.close()

    plt.plot(df['train_mse_loss'] * loss_weights[1], label='Train')
    plt.plot(df['test_mse_loss'] * loss_weights[1], label='Test')
    plt.legend()
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.xticks(ticks, [i * show_step for i in ticks], rotation=70)
    plt.savefig(DIR_RESULTS_c + '/MSEcurve.pdf', format='pdf',\
                transparent=True, dpi=200, bbox_inches='tight')
    plt.savefig(DIR_RESULTS_c + '/MSEcurve.png', format='png',\
                transparent=True, dpi=200, bbox_inches='tight')
    plt.close()