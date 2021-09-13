import pathlib

import numpy as np
import bottleneck as bn

from scipy import sparse

from src.aws_manager import AwsManager
import traceback

dir = pathlib.Path(__file__).parent.parent.absolute() / 'poiData'


class PlainUserRecommender:

    def __init__(self):
        self.aws_manager = AwsManager()
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

        return rating_matrix

    def init_predictor(self):
        self.ratingdata = self.load_data()
        num_users, num_items = self.ratingdata.shape
        self.userfactors = np.random.random_sample((num_users, self.factors)).astype(
            'float32')  # Return random floats in the half-open interval [0.0, 1.0).
        self.itemfactors = np.random.random_sample((num_items, self.factors)).astype('float32')

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

    def context_mf(self, user_id, poi_start_id=0, poi_end_id=-1):
        # predict user items
        recommended_poi_names = []
        try:
            self.init_predictor()
            # uattributes = np.zeros(self.factors)
            # attributes = self.Userattributes[str(user_id)]

            # for aid in attributes:
            # uattributes += self.userattributefactors[aid, :]
            user_id += self.max_google_user_id
            predict = np.reshape(self.ubias[user_id] + self.ibias +
                                 np.sum(np.multiply(self.userfactors[user_id, :], self.itemfactors[:, :]),
                                        axis=1,
                                        dtype=np.float32), (-1,))

            idx = bn.argpartition(-predict, 10)

            if poi_end_id == -1:
                poi_end_id = len(idx)
            idx_in_range = idx[(idx >= poi_start_id) * (idx < poi_end_id)]
            recommended_poi_indices = idx_in_range[0:10]

            for index in recommended_poi_indices:
                recommended_poi_names.append(self.restaurant_poi_id_to_name_map[index])

            return recommended_poi_names
        except:
            traceback.print_exc()
            return recommended_poi_names


