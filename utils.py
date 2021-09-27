#        Interpreting Node Embedding with Text-Labeled Graphs
    
#   File:     utils.py 
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
import os, re
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths of working directories
DIR_DATA = './data/'
DIR_PREPROCESSED_DATA = './preprocessed_data/'
DIR_TRAIN_TEST_DATA = './train_test_data/'
DIR_RESULTS = './results/'
MODEL_PATH = DIR_PREPROCESSED_DATA + '/pretrained_language_model'

# K-core dimension
K = 5

# regex to remove punctuation, special characters, numbers from a sentence
SENTENCE_SPLIT_REGEX = re.compile(r'[^a-zA-Z]')
NUM_FEATURES = 200
CATEGORIES = ['automotive', 'baby', 'beauty', 'CDs',
              'cellphones', 'digital_music', 'food', 'health',
              'instant_videos', 'kindle', 'musical_instruments', 'office',
              'patio', 'pet', 'sports', 'tools', 'toys', 'videogames']
CATEGORIES = ['patio']

STOPWORDS = set(stopwords.words('english')).union(
 {"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone",
  "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and",
  "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
  "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below",
  "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant",
  "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each",
  "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
  "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for",
  "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has",
  "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him",
  "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it",
  "its", "itself", "just", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "maybe", "me",
  "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself",
  "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not",
  "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others",
  "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather",
  "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side",
  "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes",
  "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then",
  "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin",
  "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top",
  "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was",
  "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby",
  "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom",
  "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves",
  "the"})

N_TOP_WORDS = 10
C = 2

def count_reviews(data, filter_flag=True):
    '''
    INPUT
    ----------
    data : variable containing the reviews [user_ID, product_ID, vec_review, rating]

    OUTPUT
    ----------
    dict_users : {userID : {'mean' : x, 'std' : y}}
            'mean' : average of the ratings given by the user
            'std' : standard deviation of the ratings given by the user

    dict_products : {productID : count of associated reviews}
    '''

    dict_users = {}
    dict_products = {}

    for review in data:
        reviewerID = review[0]
        asin = review[1]
        rating = review[3]

        if dict_users.get(reviewerID, None) is None:
            dict_users[reviewerID] = [rating]
        else:
            dict_users[reviewerID].append(rating)

        if dict_products.get(asin, None) is None:
            dict_products[asin] = 1
        else:
            dict_products[asin] += 1

    for user in dict_users.keys():
        ratings_u = dict_users[user]
        std_ratings_u = float(np.std(ratings_u))
        avg_ratings_u = 1.0 * sum(ratings_u) / len(ratings_u)
        dict_users[user] = {'mean': avg_ratings_u, 'std': std_ratings_u}

        if filter_flag:
            if std_ratings_u != 0:
                dict_users[user] = {'mean': avg_ratings_u, 'std': std_ratings_u}

            # remove users who gave same rating to each product
            else:
                del dict_users[user]
    return dict_users, dict_products


def mapping(users_set, products_set):
    
    '''
    INPUT
    ----------
    users_set : list/set of users' ID
    products_set : list/set of products' ID'
    
    OUTPUT
    ----------
    mapping_users : {userID : index (as integer)}    
    mapping_products : {productID : index (as integer)}
    '''
    
    count = 0
    mapping_users = {}
    mapping_products = {}
    
    for user in users_set:
        mapping_users[user] = count
        count += 1
    
    count = 0    
    for product in products_set:
        mapping_products[product] = count
        count += 1
        
    return mapping_users, mapping_products