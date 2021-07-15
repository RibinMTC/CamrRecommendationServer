import json
import pathlib

import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

from src.aws_manager import AwsManager
import traceback


class ItemSimilarityRecommender:

    def __init__(self):
        self.aws_manager = AwsManager()
        self.topK = 4
        poi_db = self.aws_manager.load_all_pois()
        jpd = pd.DataFrame(poi_db)

        cols_to_drop = ['Description', 'Gps', 'Reviews']

        jpd = jpd.drop(cols_to_drop, axis=1)

        self.df = jpd[pd.to_numeric(jpd['Rating']) > 0]
        self.df['Categories'] = self.df.Categories.apply(lambda x: x[0:-1].split(','))
        df1 = self.df['Categories'].explode()
        self.df2 = self.df.join(pd.crosstab(df1.index, df1))

        df1 = self.df2[set(self.df['Categories'].explode())].astype(str).agg(''.join, axis=1)
        df2 = pd.concat([self.df, df1], axis=1)

        self.df2 = df2.rename({0: 'representation'}, axis=1)

    def get_similar_items(self, item_id):
        new_ranked_order = []
        try:
            itemid = self.df2[self.df2.Name == item_id]['representation']
            x_list = list(itemid.values[0])

            self.df['scores'] = self.df2['representation'].apply(lambda x: cosine_similarity([list(x)], [x_list]))

            sorted_df = self.df.sort_values(by=['scores'], ascending=False)
            ranked_order = sorted_df['Name'][:self.topK].values

            new_ranked_order = np.delete(ranked_order, np.where(ranked_order == item_id))

            return new_ranked_order
        except:
            traceback.print_exc()
            return new_ranked_order

    # def __init__(self):
    #     self.topK = 4
    #     curr_file_path = pathlib.Path(__file__).parent.parent.absolute() / 'poiData'
    #
    #     jpd = pd.read_json(open(curr_file_path / 'combinedCustomPlaceDetailsDataNew.json', 'r', encoding="utf8"))
    #
    #     cols_to_drop = ['address_components', 'adr_address', 'business_status',
    #                     'formatted_address', 'formatted_phone_number', 'geometry', 'icon',
    #                     'international_phone_number', 'photos', 'place_id', 'plus_code', 'reference', 'url',
    #                     'utc_offset', 'vicinity', 'website', 'opening_hours', 'price_level', 'permanently_closed']
    #
    #     jpd = jpd.drop(cols_to_drop, axis=1)
    #
    #     self.df = jpd[jpd['rating'] > 0]
    #
    #     df1 = self.df['types'].explode()
    #     self.df2 = self.df.join(pd.crosstab(df1.index, df1))
    #
    #     df1 = self.df2[set(self.df['types'].explode())].astype(str).agg(''.join, axis=1)
    #     df2 = pd.concat([self.df, df1], axis=1)
    #
    #     self.df2 = df2.rename({0: 'representation'}, axis=1)
    #
    # def get_popular_items(self, item_in_view):
    #     topK = 5
    #     dir = os.getcwd();
    #
    #     restarants_data = json.load(open(dir + '/restaurantPlaceDetailsData.json', 'r'))
    #
    #     result = []
    #     for item in restarants_data:
    #         result.append(item['result'])
    #
    #     jpd = pd.json_normalize(result, max_level=0)
    #
    #     cols_to_drop = ['address_components', 'adr_address', 'business_status',
    #                     'formatted_address', 'formatted_phone_number', 'geometry', 'icon',
    #                     'international_phone_number', 'photos', 'place_id', 'plus_code', 'reference', 'url',
    #                     'utc_offset', 'vicinity', 'website', 'opening_hours', 'price_level', 'permanently_closed']
    #
    #     jpd = jpd.drop(cols_to_drop, axis=1)
    #
    #     df = jpd[jpd['rating'] > 0]
    #
    #     sorted_df = df.sort_values(by=['user_ratings_total'], ascending=False)
    #     names = sorted_df['name'][:topK].index
    #     ranked_order = sorted_df['name'][(sorted_df.name).isin(item_in_view)].values
    #     return ranked_order
    #
    #
    #
    # # call this funtion to get items similar to an inpurt item (pass the name of the item)
    # def get_similar_items(self, item_id):
    #     itemid = self.df2[self.df2.name == item_id]['representation']
    #     x_list = list(itemid.values[0])
    #
    #     self.df['scores'] = self.df2['representation'].apply(lambda x: cosine_similarity([list(x)], [x_list]))
    #
    #     sorted_df = self.df.sort_values(by=['scores'], ascending=False)
    #     ranked_order = sorted_df['name'][:self.topK].values
    #
    #     new_ranked_order = np.delete(ranked_order, np.where(ranked_order == item_id))
    #
    #     return new_ranked_order
