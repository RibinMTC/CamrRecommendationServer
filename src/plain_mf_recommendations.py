import json
import pathlib

import boto3
import pandas as pd
import numpy as np
from scipy import sparse
import scipy.sparse
import bottleneck as bn


dir = pathlib.Path(__file__).parent.parent.absolute() / 'poiData'



class AwsManager:
    def __init__(self):
        curr_file_path = pathlib.Path(__file__).parent.parent.absolute()
        with open(curr_file_path / 'aws_credentials.json') as aws_credentials_file:
            aws_credentials = json.load(aws_credentials_file)

        session = boto3.Session(
            aws_access_key_id=aws_credentials['aws_access_key_id'],
            aws_secret_access_key=aws_credentials['aws_secret_access_key'],
            region_name=aws_credentials['region_name']
        )

        self.dynamodb = session.resource('dynamodb')

    def scan_table(self, table_name):
        table = self.dynamodb.Table(table_name)

        response = table.scan(
        )

        return response['Items']

    def load_all_users(self):
        return self.scan_table('camr-users')

    def load_all_pois(self):
        return self.scan_table('camr-poi-storage')

    def load_all_user_prefs(self):
        return self.scan_table('camr-user-item-preferences')


def load_data():

    rating_matrix = scipy.sparse.load_npz(dir / 'ratingdatabase.npz')
    with open(dir / 'profiles.json', 'r') as read_file:
        uprofile = json.loads(read_file.read())


    aws = AwsManager()

    users = aws.load_all_users()

    pois = aws.load_all_pois()

    user_prefs = aws.load_all_user_prefs()

    vals, cols, rows = zip(*user_prefs)

    rating_matrix = sparse.csr_matrix((vals, (rows, cols)))
    #sparse.csr_matrix((user_prefs['UserPreference'].values.astype(float),(user_prefs['UserId'].values,user_prefs['PoiId'].values)))

    return uprofile, rating_matrix


class PlainUserRecommender:

    def __init__(self, user_attribute_codes_len):
        self.user_attribute_codes_len = user_attribute_codes_len
        self.factors = 50
        self.lr = 0.001
        self.reg = 0.1
        self.iterations = 50
        self.Userattributes, self.ratingdata = load_data()
        self.itemfactors = None
        self.userattributefactors = None
        self.restaurant_poi_data = pd.read_json(open(dir / 'restaurant.json', 'r'))
        self.init_predictor()

    def init_predictor(self):
        num_users, num_items = self.ratingdata.shape
        num_attributes = self.user_attribute_codes_len
        self.userfactors = np.random.random_sample((num_users, self.factors)).astype(
            'float32')  # Return random floats in the half-open interval [0.0, 1.0).
        self.itemfactors = np.random.random_sample((num_items, self.factors)).astype('float32')
        #self.userattributefactors = np.random.random_sample((num_attributes, self.factors)).astype('float32')

        self.ibias = np.zeros(num_items)
        self.ubias = np.zeros(num_users)

        # row, col = local_train_data.nonzero()
        cx = self.ratingdata.tocoo()
        # cx = sparse.coo_matrix(local_train_data)
        for iter in range(self.iterations):

            for u, i, value in zip(cx.row, cx.col, cx.data):

                #uattributes = np.zeros(self.factors)
                #attributes = self.Userattributes[str(u)]

                #for aid in attributes:
                    #uattributes += self.userattributefactors[aid, :]

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

        #uattributes = np.zeros(self.factors)
        #attributes = self.Userattributes[str(user_id)]

        #for aid in attributes:
            #uattributes += self.userattributefactors[aid, :]

        predict = np.reshape(self.ubias[user_id] + self.ibias +
                             np.sum(np.multiply(self.userfactors[user_id, :], self.itemfactors[:, :]),
                                    axis=1,
                                    dtype=np.float32), (-1,))

        idx = bn.argpartition(-predict, 10)

        recommended_poi_indices = idx[0:5]
        recommended_poi_names = []
        for index in recommended_poi_indices:
            recommended_poi_names.append(self.restaurant_poi_data["name"][index])

        return recommended_poi_names
