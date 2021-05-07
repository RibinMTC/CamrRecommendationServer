import json
import pathlib

curr_file_path = pathlib.Path(__file__).parent.parent.absolute() / 'poiData'

new_poi_dict = {}
with open(curr_file_path / 'combinedCustomPlaceDetailsData.json', 'r', encoding="utf8") as poi_json:
    poi_data = json.load(poi_json)
    for single_poi in poi_data:
        if 'Cuisine' in single_poi.keys():
            cuisine = single_poi['Cuisine']
            if cuisine is None:
                continue
            cuisine_list = [x.strip() for x in cuisine.split(",")]
            combined_types = single_poi['types'] + cuisine_list
            single_poi['types'] = combined_types


with open(curr_file_path / 'combinedCustomPlaceDetailsDataNew.json', 'w', encoding="utf8") as poi_json:
    json.dump(poi_data, poi_json)

