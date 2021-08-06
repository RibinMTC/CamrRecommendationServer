import json
import pathlib

import pandas as pd
import numpy as np
from pandas.core.nanops import bn
from scipy import sparse

from src.aws_manager import AwsManager

dir = pathlib.Path(__file__).parent.parent.absolute() / 'poiData'


class UserContextBasedRecommender:

    def __init__(self, user_attribute_codes_len):
        self.aws_manager = AwsManager()
        self.user_attribute_codes_len = user_attribute_codes_len
        self.factors = 50
        self.lr = 0.001
        self.reg = 0.1
        self.iterations = 50
        self.Userattributes, self.ratingdata = self.load_data()
        self.itemfactors = None
        self.userattributefactors = None
        pois_db = self.aws_manager.load_all_pois()
        self.restaurant_poi_id_to_name_map = {}
        self.old_user_likes = None
        for poi in pois_db:
            self.restaurant_poi_id_to_name_map[int(poi['PoiId']) - 1] = poi['Name']
        self.init_predictor()

    def load_data(self):

        google_user_prefs = self.aws_manager.load_all_google_user_prefs()
        real_user_prefs = self.aws_manager.load_all_real_user_prefs()
        ratings = []
        users = []
        pois = []
        for user_pref in google_user_prefs:
            users.append(int(user_pref['UserId']) - 1)
            pois.append(int(user_pref['PoiId']) - 1)
            ratings.append(int(float(user_pref['UserPreference'])))

        self.max_google_user_id = max(users)
        for user_pref in real_user_prefs:
            users.append(self.max_google_user_id + int(user_pref['UserId']))
            pois.append(int(user_pref['PoiId']) - 1)
            ratings.append(int(float(user_pref['UserPreference'])))

        rating_matrix = sparse.csr_matrix((ratings, (users, pois)))

        user_attributes = self.get_real_user_attributes()
        return user_attributes, rating_matrix

    def get_real_user_attributes(self):
        user_attributes = {}
        users = self.aws_manager.load_all_users()
        for user in users:
            user_id = int(user['UserId'])
            attribute_num = []
            age = int(user['Age'])
            attribute_num.append(self.age_to_attribute_num_mapper(age))
            gender = user['Gender']
            attribute_num.append(self.gender_to_attribute_num_mapper(gender))
            personality_traits = user['PersonalityTraits']
            agreeableness = int(personality_traits['Agreeable'])
            calmness = int(personality_traits['Calm'])
            dependability = int(personality_traits['Dependable'])
            extraversion = int(personality_traits['Extraverted'])
            openness = int(personality_traits['Open'])

            offset = 7
            attribute_num.append(self.personality_trait_to_attribute_num_mapper(agreeableness) + offset)
            offset += 5
            attribute_num.append(self.personality_trait_to_attribute_num_mapper(calmness) + offset)
            offset += 5
            attribute_num.append(self.personality_trait_to_attribute_num_mapper(dependability) + offset)
            offset += 5
            attribute_num.append(self.personality_trait_to_attribute_num_mapper(extraversion) + offset)
            offset += 5
            attribute_num.append(self.personality_trait_to_attribute_num_mapper(openness) + offset)

            user_attributes[self.max_google_user_id + user_id] = attribute_num

        return user_attributes

    def age_to_attribute_num_mapper(self, age):
        if age < 18:
            return 0
        elif age < 26:
            return 1
        elif age < 40:
            return 2
        elif age < 55:
            return 3
        elif age < 70:
            return 4
        else:
            return 5

    def gender_to_attribute_num_mapper(self, gender):
        if gender == 'male':
            return 6
        elif gender == 'female':
            return 7
        else:
            print("Error. Gender not recognized")
            return -1

    def personality_trait_to_attribute_num_mapper(self, attribute):
        attribute_percentage = attribute / 7.0 * 100.0
        if attribute_percentage < 20:
            return 1
        elif attribute_percentage < 40:
            return 2
        elif attribute_percentage < 60:
            return 3
        elif attribute_percentage < 80:
            return 4
        else:
            return 5

    def init_predictor(self):
        num_users, num_items = self.ratingdata.shape
        num_attributes = self.user_attribute_codes_len
        self.userfactors = np.random.random_sample((num_users, self.factors)).astype(
            'float32')  # Return random floats in the half-open interval [0.0, 1.0).
        self.itemfactors = np.random.random_sample((num_items, self.factors)).astype('float32')
        self.userattributefactors = np.random.random_sample((num_attributes, self.factors)).astype('float32')

        self.ibias = np.zeros(num_items)
        self.ubias = np.zeros(num_users)

        # row, col = local_train_data.nonzero()
        cx = self.ratingdata.tocoo()
        # cx = sparse.coo_matrix(local_train_data)
        for iter in range(self.iterations):

            for u, i, value in zip(cx.row, cx.col, cx.data):

                uattributes = np.zeros(self.factors)
                attributes = []
                if u in self.Userattributes.keys():
                    attributes = self.Userattributes[u]

                for aid in attributes:
                    uattributes += self.userattributefactors[aid, :]

                predict = self.ubias[u] + self.ibias[i] + np.dot((self.userfactors[u, :] + uattributes[:]),
                                                                 self.itemfactors[i, :])

                err = (value - predict)

                self.ubias[u] += self.lr * (err - self.reg * self.ubias[u])
                self.ibias[i] += self.lr * (err - self.reg * self.ibias[i])

                uf = self.userfactors[u, :]
                itf = self.itemfactors[i, :]

                d = (err * self.itemfactors[i, :] - self.reg * uf)
                self.userfactors[u, :] += self.lr * d

                d = (err * (self.userfactors[u, :] + uattributes[:]) - self.reg * itf)
                self.itemfactors[i, :] += self.lr * d

                for aid in attributes:
                    d = (err * itf - self.reg * self.userattributefactors[aid, :])
                    self.userattributefactors[aid, :] += self.lr * d

    def additional_preference_training(self, user_id):

        new_user_likes = self.aws_manager.load_all_real_user_likes(user_id)
        if self.old_user_likes == new_user_likes:
            return

        self.old_user_likes = new_user_likes
        for iter in range(self.iterations):

            for user_pref in new_user_likes:
                u = int(user_pref['UserId'])
                i = int(user_pref['PoiId'])
                value = int(user_pref['UserPreference'])

                # for u, i, value in zip(new_preferences[0], new_preferences[1], new_preferences[2]):

                uattributes = np.zeros(self.factors)
                attributes = []
                if u in self.Userattributes.keys():
                    attributes = self.Userattributes[u]

                for aid in attributes:
                    uattributes += self.userattributefactors[aid, :]

                predict = self.ubias[u] + self.ibias[i] + np.dot((self.userfactors[u, :] + uattributes[:]),
                                                                 self.itemfactors[i, :])

                err = (value - predict)

                self.ubias[u] += 5 * self.lr * (err - self.reg * self.ubias[u])
                self.ibias[i] += 5 * self.lr * (err - self.reg * self.ibias[i])

                uf = self.userfactors[u, :]
                itf = self.itemfactors[i, :]

                d = (err * self.itemfactors[i, :] - self.reg * uf)
                self.userfactors[u, :] += 5 * self.lr * d

                d = (err * (self.userfactors[u, :] + uattributes[:]) - self.reg * itf)
                self.itemfactors[i, :] += 5 * self.lr * d

                for aid in attributes:
                    d = (err * (itf) - self.reg * self.userattributefactors[aid, :])
                    self.userattributefactors[aid, :] += 5 * self.lr * d

    def context_mf(self, user_id, poi_start_id=0, poi_end_id=-1):
        # predict user items

        uattributes = np.zeros(self.factors)
        attributes = self.Userattributes[user_id+ self.max_google_user_id]

        for aid in attributes:
            uattributes += self.userattributefactors[aid, :]

        predict = np.reshape(self.ubias[user_id] + self.ibias +
                             np.sum(np.multiply(self.userfactors[user_id, :] + uattributes[:], self.itemfactors[:, :]),
                                    axis=1,
                                    dtype=np.float32), (-1,))

        idx = bn.argpartition(-predict, 10)

        if poi_end_id == -1:
            poi_end_id = len(idx)
        idx_in_range = idx[(idx >= poi_start_id) * (idx < poi_end_id)]

        recommended_poi_indices = idx_in_range[0:5]
        recommended_poi_names = []
        for index in recommended_poi_indices:
            recommended_poi_names.append(self.restaurant_poi_id_to_name_map[index])

        return recommended_poi_names
