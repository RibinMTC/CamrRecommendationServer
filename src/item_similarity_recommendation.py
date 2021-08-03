import pandas as pd
import numpy as np
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

    def get_similar_items(self, item_id, poi_start_id=0, poi_end_id=-1):
        new_ranked_order = []
        try:
            itemid = self.df2[self.df2.Name == item_id]['representation']
            if len(itemid.values) == 0:
                print("Poi with name: " + item_id + " not found! Aborting similar items recommendation")
                return new_ranked_order
            x_list = list(itemid.values[0])

            self.df['scores'] = self.df2['representation'].apply(lambda x: cosine_similarity([list(x)], [x_list]))

            sorted_df = self.df.sort_values(by=['scores'], ascending=False)

            if poi_end_id == -1:
                poi_end_id = len(sorted_df.PoiId)
            sorted_df = sorted_df[(poi_start_id <= sorted_df.PoiId) * (sorted_df.PoiId < poi_end_id)]

            ranked_order = sorted_df['Name'][:self.topK].values

            new_ranked_order = np.delete(ranked_order, np.where(ranked_order == item_id))

            return new_ranked_order
        except:
            traceback.print_exc()
            return new_ranked_order
