import json
import pathlib

import pandas as pd
import numpy as np
import bottleneck as bn

from scipy import sparse

from src.aws_manager import AwsManager
import traceback

dir = pathlib.Path(__file__).parent.parent.absolute() / 'poiData'


class PlainUserRecommender:

    def __init__(self, user_attribute_codes_len):
        self.aws_manager = AwsManager()
        self.user_attribute_codes_len = user_attribute_codes_len
        self.factors = 50
        self.lr = 0.001
        self.reg = 0.1
        self.iterations = 50
        self.itemfactors = None
        self.userattributefactors = None
        pois_db = self.aws_manager.load_all_pois()
        self.restaurant_poi_id_to_name_map = {}
        for poi in pois_db:
            self.restaurant_poi_id_to_name_map[int(poi['PoiId']) - 1] = poi['Name']

        self.init_predictor()

    def load_data(self):
        with open(dir / 'profiles.json', 'r') as read_file:
            uprofile = json.loads(read_file.read())

        user_prefs = self.aws_manager.load_all_user_prefs()

        ratings = []
        users = []
        pois = []
        for user_pref in user_prefs:
            users.append(int(user_pref['UserId']) - 1)
            pois.append(int(user_pref['PoiId']) - 1)
            ratings.append(int(float(user_pref['UserPreference'])))

        rating_matrix = sparse.csr_matrix((ratings, (users, pois)))

        return uprofile, rating_matrix

    def init_predictor(self):
        self.Userattributes, self.ratingdata = self.load_data()
        num_users, num_items = self.ratingdata.shape
        num_attributes = self.user_attribute_codes_len
        self.userfactors = np.random.random_sample((num_users, self.factors)).astype(
            'float32')  # Return random floats in the half-open interval [0.0, 1.0).
        self.itemfactors = np.random.random_sample((num_items, self.factors)).astype('float32')
        # self.userattributefactors = np.random.random_sample((num_attributes, self.factors)).astype('float32')

        self.ibias = np.zeros(num_items)
        self.ubias = np.zeros(num_users)

        # row, col = local_train_data.nonzero()
        cx = self.ratingdata.tocoo()
        # cx = sparse.coo_matrix(local_train_data)
        for iteration in range(self.iterations):

            for u, i, value in zip(cx.row, cx.col, cx.data):
                # uattributes = np.zeros(self.factors)
                # attributes = self.Userattributes[str(u)]

                # for aid in attributes:
                # uattributes += self.userattributefactors[aid, :]

                predict = self.ubias[u] + self.ibias[i] + np.dot((self.userfactors[u, :]),
                                                                 self.itemfactors[i, :])

                err = (value - predict)

                self.ubias[u] += self.lr * (err - self.reg * self.ubias[u])
                self.ibias[i] += self.lr * (err - self.reg * self.ibias[i])

                uf = self.userfactors[u, :]
                itf = self.itemfactors[i, :]

                d = (err * self.itemfactors[i, :] - self.reg * uf)
                self.userfactors[u, :] += self.lr * d

                d = (err * (self.userfactors[u, :]) - self.reg * itf)
                self.itemfactors[i, :] += self.lr * d

    def context_mf(self, user_id):
        # predict user items
        recommended_poi_names = []
        try:
            self.init_predictor()
            # uattributes = np.zeros(self.factors)
            # attributes = self.Userattributes[str(user_id)]

            # for aid in attributes:
            # uattributes += self.userattributefactors[aid, :]
            user_id -= 1
            predict = np.reshape(self.ubias[user_id] + self.ibias +
                                 np.sum(np.multiply(self.userfactors[user_id, :], self.itemfactors[:, :]),
                                        axis=1,
                                        dtype=np.float32), (-1,))

            idx = bn.argpartition(-predict, 10)

            recommended_poi_indices = idx[0:5]

            for index in recommended_poi_indices:
                recommended_poi_names.append(self.restaurant_poi_id_to_name_map[index])

            return recommended_poi_names
        except:
            traceback.print_exc()
            return recommended_poi_names

# class UserNoContextRecommender:
#
#     def __init__(self, user_attribute_codes_len):
#         self.aws_manager = AwsManager()
#         self.user_attribute_codes_len = user_attribute_codes_len
#         self.factors = 50
#         self.lr = 0.001
#         self.reg = 0.1
#         self.iterations = 50
#         self.Userattributes, self.ratingdata = self.load_data()
#         self.itemfactors = None
#         self.userattributefactors = None
#         self.restaurant_poi_data = pd.read_json(open(dir / 'restaurant.json', 'r'))
#         self.init_predictor()
#
#     def load_data(self):
#         with open(dir / 'profiles.json', 'r') as read_file:
#             uprofile = json.loads(read_file.read())
#
#         user_prefs = self.aws_manager.load_all_user_prefs()
#
#         ratings = []
#         users = []
#         pois = []
#         for user_pref in user_prefs:
#             users.append(int(user_pref['UserId']))
#             pois.append(int(user_pref['PoiId']))
#             ratings.append(int(float(user_pref['UserPreference'])))
#
#         rating_matrix = sparse.csr_matrix((ratings, (users, ratings)))
#
#         return uprofile, rating_matrix
#
#     def init_predictor(self):
#         num_users, num_items = self.ratingdata.shape
#         num_attributes = self.user_attribute_codes_len
#         self.userfactors = np.random.random_sample((num_users, self.factors)).astype(
#             'float32')  # Return random floats in the half-open interval [0.0, 1.0).
#         self.itemfactors = np.random.random_sample((num_items, self.factors)).astype('float32')
#         self.userattributefactors = np.random.random_sample((num_attributes, self.factors)).astype('float32')
#
#         self.ibias = np.zeros(num_items)
#         self.ubias = np.zeros(num_users)
#
#         # row, col = local_train_data.nonzero()
#         cx = self.ratingdata.tocoo()
#         # cx = sparse.coo_matrix(local_train_data)
#         for iter in range(self.iterations):
#
#             for u, i, value in zip(cx.row, cx.col, cx.data):
#
#                 uattributes = np.zeros(self.factors)
#                 attributes = self.Userattributes[str(u)]
#
#                 for aid in attributes:
#                     uattributes += self.userattributefactors[aid, :]
#
#                 predict = self.ubias[u] + self.ibias[i] + np.dot((self.userfactors[u, :] + uattributes[:]),
#                                                                  self.itemfactors[i, :])
#
#                 err = (value - predict)
#
#                 self.ubias[u] += self.lr * (err - self.reg * self.ubias[u])
#                 self.ibias[i] += self.lr * (err - self.reg * self.ibias[i])
#
#                 uf = self.userfactors[u, :]
#                 itf = self.itemfactors[i, :]
#
#                 d = (err * self.itemfactors[i, :] - self.reg * uf)
#                 self.userfactors[u, :] += self.lr * d
#
#                 d = (err * (self.userfactors[u, :] + uattributes[:]) - self.reg * itf)
#                 self.itemfactors[i, :] += self.lr * d
#
#                 for aid in attributes:
#                     d = (err * (itf) - self.reg * self.userattributefactors[aid, :])
#                     self.userattributefactors[aid, :] += self.lr * d
#
#     def additional_preference_training(self, new_preferences):
#
#         for iter in range(self.iterations):
#
#             for user_pref in new_preferences:
#                 u = int(user_pref['UserId'])
#                 i = int(user_pref['PoiId'])
#                 value = int(user_pref['UserPreference'])
#
#                 # for u, i, value in zip(new_preferences[0], new_preferences[1], new_preferences[2]):
#
#                 uattributes = np.zeros(self.factors)
#                 attributes = self.Userattributes[str(u)]
#
#                 for aid in attributes:
#                     uattributes += self.userattributefactors[aid, :]
#
#                 predict = self.ubias[u] + self.ibias[i] + np.dot((self.userfactors[u, :] + uattributes[:]),
#                                                                  self.itemfactors[i, :])
#
#                 err = (value - predict)
#
#                 self.ubias[u] += 5 * self.lr * (err - self.reg * self.ubias[u])
#                 self.ibias[i] += 5 * self.lr * (err - self.reg * self.ibias[i])
#
#                 uf = self.userfactors[u, :]
#                 itf = self.itemfactors[i, :]
#
#                 d = (err * self.itemfactors[i, :] - self.reg * uf)
#                 self.userfactors[u, :] += 5 * self.lr * d
#
#                 d = (err * (self.userfactors[u, :] + uattributes[:]) - self.reg * itf)
#                 self.itemfactors[i, :] += 5 * self.lr * d
#
#                 for aid in attributes:
#                     d = (err * (itf) - self.reg * self.userattributefactors[aid, :])
#                     self.userattributefactors[aid, :] += 5 * self.lr * d
#
#     def context_mf(self, user_id):
#         # predict user items
#
#         uattributes = np.zeros(self.factors)
#         attributes = self.Userattributes[str(user_id)]
#
#         for aid in attributes:
#             uattributes += self.userattributefactors[aid, :]
#
#         predict = np.reshape(self.ubias[user_id] + self.ibias +
#                              np.sum(np.multiply(self.userfactors[user_id, :] + uattributes[:], self.itemfactors[:, :]),
#                                     axis=1,
#                                     dtype=np.float32), (-1,))
#
#         idx = bn.argpartition(-predict, 10)
#
#         recommended_poi_indices = idx[0:5]
#         recommended_poi_names = []
#         for index in recommended_poi_indices:
#             recommended_poi_names.append(self.restaurant_poi_data["name"][index])
#
#         return recommended_poi_names
