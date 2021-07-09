import json
import pathlib

import pandas as pd


def get_popular_items(item_in_view):
    topK = 5

    curr_file_path = pathlib.Path(__file__).parent.parent.absolute() / 'poiData'

    restarants_data = json.load(open(curr_file_path / 'restaurant.json', 'r', encoding="utf8"))

    result = []
    for item in restarants_data['results']:
        result.append(item['result'])

    jpd = pd.json_normalize(result, max_level=0)

    cols_to_drop = ['address_components', 'adr_address', 'business_status',
                    'formatted_address', 'formatted_phone_number', 'geometry', 'icon',
                    'international_phone_number', 'photos', 'place_id', 'plus_code', 'reference', 'url',
                    'utc_offset', 'vicinity', 'website', 'opening_hours', 'price_level', 'permanently_closed']

    jpd = jpd.drop(cols_to_drop, axis=1)

    df = jpd[jpd['rating'] > 0]

    sorted_df = df.sort_values(by=['user_ratings_total'], ascending=False)
    #names = sorted_df['name'][:topK].index
    ranked_order = sorted_df['name'][(sorted_df.name).isin(item_in_view)].values
    return ranked_order
