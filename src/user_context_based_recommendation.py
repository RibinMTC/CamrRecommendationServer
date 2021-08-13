import copy
import pathlib

import numpy as np
from pandas.core.nanops import bn
from scipy import sparse

from src.aws_manager import AwsManager
from src.usercontext_database_creator import userattributescodes

dir = pathlib.Path(__file__).parent.parent.absolute() / 'poiData'


class TrainingData:

    def __init__(self, num_users, num_items, factors, num_attributes):
        self.userfactors = np.random.random_sample((num_users, factors)).astype(
            'float32')
        self.itemfactors = np.random.random_sample((num_items, factors)).astype('float32')
        self.userattributefactors = np.random.random_sample((num_attributes, factors)).astype('float32')

        self.ibias = np.zeros(num_items)
        self.ubias = np.zeros(num_users)


class UserAttributesLoader:

    def __init__(self):
        self.aws_manager = AwsManager()

    def get_real_user_attributes(self, user_id_offset):
        user_attributes = {}
        users = self.aws_manager.load_all_users()
        for user in users:
            user_id = user_id_offset + int(user['UserId'])
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

            user_attributes[user_id] = attribute_num

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


class UserContextBasedRecommender:

    def __init__(self):
        self.aws_manager = AwsManager()
        self.factors = 50
        self.lr = 0.001
        self.reg = 0.1
        self.iterations = 50
        self.Userattributes, self.ratingdata = self.load_data()
        pois_db = self.aws_manager.load_all_pois()
        self.restaurant_poi_id_to_name_map = {}
        self.old_user_likes = None
        self.old_training_data = None
        for poi in pois_db:
            self.restaurant_poi_id_to_name_map[int(poi['PoiId']) - 1] = poi['Name']
        self.user_attribute_codes_len = len(userattributescodes)
        self.not_placed_pois = [36, 39]  # delish not placed. Enzian-vegane bäckerei does not exist.
        self.old_user_id = -1
        self.initial_training_data = None
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

        user_attributes_loader = UserAttributesLoader()
        user_attributes = user_attributes_loader.get_real_user_attributes(self.max_google_user_id)
        return user_attributes, rating_matrix

    def init_predictor(self):
        num_users, num_items = self.ratingdata.shape
        num_attributes = self.user_attribute_codes_len
        self.initial_training_data = TrainingData(num_users, num_items, self.factors, num_attributes)

        cx = self.ratingdata.tocoo()
        self.train_predictor(sparse_matrix_coo=cx, training_data=self.initial_training_data)

    def additional_preference_training(self, new_user_likes):

        if self.old_user_likes == new_user_likes:
            print("Preferences did not change. Skipping training of predictor")
            return self.old_training_data
        self.old_user_likes = new_user_likes

        new_training_data = copy.deepcopy(self.initial_training_data)
        cx = self.get_updated_rating_matrix(new_user_likes)
        self.train_predictor(sparse_matrix_coo=cx.tocoo(), training_data=new_training_data, multiplier=2)
        self.old_training_data = new_training_data
        return new_training_data

    def get_updated_rating_matrix(self, new_user_likes):
        cx = self.ratingdata.tolil(copy=True)
        for user_pref in new_user_likes:
            u = int(user_pref['UserId']) + self.max_google_user_id
            i = int(user_pref['PoiId']) - 1
            value = int(user_pref['UserPreference'])

            cx[u, i] = value
        return cx

    def train_predictor(self, sparse_matrix_coo, training_data, multiplier=1):
        for iter in range(self.iterations):

            for u, i, value in zip(sparse_matrix_coo.row, sparse_matrix_coo.col, sparse_matrix_coo.data):

                uattributes = np.zeros(self.factors)
                attributes = []
                if u in self.Userattributes.keys():
                    attributes = self.Userattributes[u]

                for aid in attributes:
                    uattributes += training_data.userattributefactors[aid, :]

                predict = training_data.ubias[u] + training_data.ibias[i] + np.dot(
                    (training_data.userfactors[u, :] + uattributes[:]),
                    training_data.itemfactors[i, :])

                err = (value - predict)

                training_data.ubias[u] += multiplier * self.lr * (err - self.reg * training_data.ubias[u])
                training_data.ibias[i] += multiplier * self.lr * (err - self.reg * training_data.ibias[i])

                uf = training_data.userfactors[u, :]
                itf = training_data.itemfactors[i, :]

                d = (err * training_data.itemfactors[i, :] - self.reg * uf)
                training_data.userfactors[u, :] += multiplier * self.lr * d

                d = (err * (training_data.userfactors[u, :] + uattributes[:]) - self.reg * itf)
                training_data.itemfactors[i, :] += multiplier * self.lr * d

                for aid in attributes:
                    d = (err * itf - self.reg * training_data.userattributefactors[aid, :])
                    training_data.userattributefactors[aid, :] += multiplier * self.lr * d

    def context_mf(self, user_id, poi_start_id=0, poi_end_id=-1):

        new_user_likes = self.aws_manager.load_all_real_user_likes(user_id)
        new_training_data = self.additional_preference_training(new_user_likes)

        user_id += self.max_google_user_id
        liked_pois = []
        for user_pref in new_user_likes:
            i = int(user_pref['PoiId']) - 1
            liked_pois.append(i)
        uattributes = np.zeros(self.factors)
        attributes = self.Userattributes[user_id]

        for aid in attributes:
            uattributes += new_training_data.userattributefactors[aid, :]

        predict = np.reshape(new_training_data.ubias[user_id] + new_training_data.ibias +
                             np.sum(np.multiply(new_training_data.userfactors[user_id, :] + uattributes[:],
                                                new_training_data.itemfactors[:, :]),
                                    axis=1,
                                    dtype=np.float32), (-1,))

        idx = bn.argpartition(-predict, 10)

        if poi_end_id == -1:
            poi_end_id = len(idx)
        idx_in_range = idx[(idx >= poi_start_id) * (idx < poi_end_id)]
        not_likes_idx = []
        for index in idx_in_range:
            if index in liked_pois or index in self.not_placed_pois:
                continue
            not_likes_idx.append(index)

        recommended_poi_indices = not_likes_idx[0:5]
        recommended_poi_names = []
        for index in recommended_poi_indices:
            recommended_poi_names.append(self.restaurant_poi_id_to_name_map[index])

        return recommended_poi_names
