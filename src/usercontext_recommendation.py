import json
import pathlib

import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import random
import csv
import scipy.sparse
import bottleneck as bn

dir = pathlib.Path(__file__).parent.parent.absolute() / 'poiData'


def GenerataUserProfile():
    gendergroup = ['male', 'female']
    agegroup = ['<18', '18-25', '26-39', '40-54', '55-69', '70+']
    opennessgroup = ['openness1', 'openness2', 'openness3', 'openness4', 'openness5']
    conscientiousnessgroup = ['conscientiousness1', 'conscientiousness2', 'conscientiousness3', 'conscientiousness4',
                              'conscientiousness5']
    extraversiongroup = ['extraversion1', 'extraversion2', 'extraversion3', 'extraversion4', 'extraversion5']
    agreeablenessgroup = ['agreeableness1', 'agreeableness2', 'agreeableness3', 'agreeableness4', 'agreeableness5']
    neuroticismgroup = ['neuroticism1', 'neuroticism2', 'neuroticism3', 'neuroticism4', 'neuroticism5']

    attributesset = list()
    attributesset.append(np.random.choice(gendergroup))
    attributesset.append(np.random.choice(agegroup))
    attributesset.append(np.random.choice(opennessgroup))
    attributesset.append(np.random.choice(conscientiousnessgroup))
    attributesset.append(np.random.choice(extraversiongroup))
    attributesset.append(np.random.choice(agreeablenessgroup))
    attributesset.append(np.random.choice(neuroticismgroup))

    idset = list()

    for ids in attributesset:
        idset.append(userattributescodes[ids])

    return idset


def CreatUsersProfiles():
    jpd = pd.read_json(open(dir / 'restaurant.json', 'r'))

    cols_to_drop = ['address_components', 'adr_address', 'business_status',
                    'formatted_address', 'formatted_phone_number', 'geometry', 'icon',
                    'international_phone_number', 'photos', 'place_id', 'plus_code', 'reference', 'url',
                    'utc_offset', 'vicinity', 'website', 'opening_hours', 'price_level', 'permanently_closed']

    jpd = jpd.drop(cols_to_drop, axis=1)

    df = jpd[jpd['rating'] > 0]

    dfratings = df[['name', 'reviews']]

    dfrate = (dfratings.explode('reviews')).reset_index().rename(columns={'index': 'itemid'})
    sfreviews = pd.json_normalize(dfrate['reviews'])

    dfratings = pd.concat([dfrate, sfreviews], axis=1)

    item_user_data = sparse.csr_matrix((dfratings['rating'].values.astype(float),
                                        (dfratings.index.values,
                                         dfratings['itemid'].values)))

    itemid = np.unique(item_user_data.indices)

    # populating user attributes
    profiles = dict()
    for u in dfratings.index.values:
        uidset = GenerataUserProfile()
        profiles[str(u)] = uidset
    # dictdf =  json.dumps(profiles)
    with open(dir / 'profiles.json', "w") as outfile:
        json.dump(profiles, outfile)

    # populating ratings database
    for u in range(len(item_user_data.indptr) - 1):
        s = random.choice((3, 5, 6, 8, 12))
        for i in range(s):
            r = random.choice((3, 4, 5))
            item_user_data[u, i] = r

    scipy.sparse.save_npz(dir / 'ratingdatabase', item_user_data)

    return 0


factors = 50
lr = 0.001
reg = 0.1
iterations = 50

userattributescodes = {}

userattributescodes['<18'] = 0
userattributescodes['18-25'] = 1
userattributescodes['26-39'] = 2
userattributescodes['40-54'] = 3
userattributescodes['55-69'] = 4
userattributescodes['70+'] = 5
userattributescodes['male'] = 6
userattributescodes['female'] = 7
userattributescodes['openness1'] = 8
userattributescodes['openness2'] = 9
userattributescodes['openness3'] = 10
userattributescodes['openness4'] = 11
userattributescodes['openness5'] = 12

userattributescodes['conscientiousness1'] = 13
userattributescodes['conscientiousness2'] = 14
userattributescodes['conscientiousness3'] = 15
userattributescodes['conscientiousness4'] = 16
userattributescodes['conscientiousness5'] = 17

userattributescodes['extraversion1'] = 18
userattributescodes['extraversion2'] = 19
userattributescodes['extraversion3'] = 20
userattributescodes['extraversion4'] = 21
userattributescodes['extraversion5'] = 22

userattributescodes['agreeableness1'] = 23
userattributescodes['agreeableness2'] = 24
userattributescodes['agreeableness3'] = 25
userattributescodes['agreeableness4'] = 26
userattributescodes['agreeableness5'] = 27

userattributescodes['neuroticism1'] = 28
userattributescodes['neuroticism2'] = 29
userattributescodes['neuroticism3'] = 30
userattributescodes['neuroticism4'] = 31
userattributescodes['neuroticism5'] = 32


def load_data():
    rating_matrix = scipy.sparse.load_npz(dir / 'ratingdatabase.npz')
    with open(dir / 'profiles.json', 'r') as read_file:
        uprofile = json.loads(read_file.read())

    return uprofile, rating_matrix


def context_mf(user_id):
    Userattributes, ratingdata = load_data()

    average = np.average(ratingdata.data)
    num_users, num_items = ratingdata.shape
    num_attributes = len(userattributescodes)
    userfactors = np.random.random_sample((num_users, factors)).astype(
        'float32')  # Return random floats in the half-open interval [0.0, 1.0).
    itemfactors = np.random.random_sample((num_items, factors)).astype('float32')
    userattributefactors = np.random.random_sample((num_attributes, factors)).astype('float32')

    ibias = np.zeros(num_items)
    ubias = np.zeros(num_users)

    # row, col = local_train_data.nonzero()
    cx = ratingdata.tocoo()
    # cx = sparse.coo_matrix(local_train_data)
    for iter in range(iterations):

        for u, i, value in zip(cx.row, cx.col, cx.data):

            uattributes = np.zeros(factors)
            attributes = Userattributes[str(u)]

            for aid in attributes:
                uattributes += userattributefactors[aid, :]

            predict = ubias[u] + ibias[i] + np.dot((userfactors[u, :] + uattributes[:]), itemfactors[i, :])

            err = (value - predict)

            ubias[u] += lr * (err - reg * ubias[u])
            ibias[i] += lr * (err - reg * ibias[i])

            uf = userfactors[u, :]
            itf = itemfactors[i, :]

            d = (err * itemfactors[i, :] - reg * uf)
            userfactors[u, :] += lr * d

            d = (err * (userfactors[u, :] + uattributes[:]) - reg * itf)
            itemfactors[i, :] += lr * d

            for aid in attributes:
                d = (err * (itf) - reg * userattributefactors[aid, :])
                userattributefactors[aid, :] += lr * d

    # predict user items

    uattributes = np.zeros(factors)
    attributes = Userattributes[str(user_id)]

    for aid in attributes:
        uattributes += userattributefactors[aid, :]

    predict = np.reshape(ubias[u] + ibias +
                         np.sum(np.multiply(userfactors[u, :] + uattributes[:], itemfactors[:, :]),
                                axis=1,
                                dtype=np.float32), (-1,))

    idx = bn.argpartition(-predict, 15)

    return idx[10:15]
