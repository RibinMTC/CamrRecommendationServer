# import main Flask class and request object
from flask import Flask, request, jsonify

# create the Flask app
from src.item_similarity_recommendation import ItemSimilarityRecommender
from src.popular_restaurants_recommendation import get_popular_items
from src.user_context_based_recommendation import UserContextBasedRecommender
from src.user_plain_recommendation import PlainUserRecommender

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World'


@app.route('/recommendation-query', methods=['GET', 'POST'])
def recommendation_query():
    recommendation = {"poiNames": []}
    try:
        request_data = request.get_json()

        ranked_pois = get_popular_items(request_data['poiNames']).tolist()
        recommendation["poiNames"] = ranked_pois

        return jsonify(recommendation)

    except Exception as e:
        print(e)
        return recommendation


@app.route('/item-similarity-recommendation-query', methods=['GET', 'POST'])
def item_similarity_recommendation_query():
    recommendation = {"poiNames": []}
    try:
        request_data = request.get_json()

        last_seen_poi = request_data['poiId']
        phase = int(request_data['phase'])
        if phase == 1:
            similar_pois = itemSimilarityRecommender.get_similar_items(last_seen_poi,
                                                                       poi_end_id=phase_separating_poi_id).tolist()
        elif phase == 2:
            similar_pois = itemSimilarityRecommender.get_similar_items(last_seen_poi,
                                                                       poi_start_id=phase_separating_poi_id).tolist()
        else:
            print("Phase not recognized. Aborting Recommendation")
            return jsonify(recommendation)

        recommendation["poiNames"] = similar_pois
        return jsonify(recommendation)

    except Exception as e:
        print(e)
        return recommendation


@app.route('/user-context-recommendation-query', methods=['GET', 'POST'])
def user_context_recommendation_query():
    recommendation = {"poiNames": []}
    try:
        request_data = request.get_json()

        user_id = int(request_data['userId'])

        phase = int(request_data['phase'])
        if phase == 1:
            recommended_pois = usercontextBasedRecommender.context_mf(user_id, poi_end_id=phase_separating_poi_id)
        elif phase == 2:
            recommended_pois = usercontextBasedRecommender.context_mf(user_id, poi_start_id=phase_separating_poi_id)
        else:
            print("Phase not recognized. Aborting Recommendation")
            return jsonify(recommendation)

        recommendation["poiNames"] = recommended_pois

        return jsonify(recommendation)

    except Exception as e:
        print(e)
        return recommendation


@app.route('/user-plain-recommendation-query', methods=['GET', 'POST'])
def user_plain_recommendation_query():
    recommendation = {"poiNames": []}
    try:
        request_data = request.get_json()

        user_id = int(request_data['userId'])
        phase = int(request_data['phase'])
        if phase == 1:
            recommended_pois = plainUserRecommender.context_mf(user_id, poi_end_id=phase_separating_poi_id)
        elif phase == 2:
            recommended_pois = plainUserRecommender.context_mf(user_id, poi_start_id=phase_separating_poi_id)
        else:
            print("Phase not recognized. Aborting Recommendation")
            return jsonify(recommendation)

        recommendation["poiNames"] = recommended_pois
        return jsonify(recommendation)

    except Exception as e:
        print(e)
        return recommendation


def get_similar_items_test(poi_name, poi_start=0, poi_end=-1):
    # test = get_popular_items( ["Bierhalle Wolf", "Rheinfelder Bierhalle", "Chop-Stick Restaurant", "Walliser Keller
    # \"The SwissRestaurant\""]) print(test)

    test = itemSimilarityRecommender.get_similar_items(poi_name, poi_start, poi_end)
    print("Similar restaurants to " + poi_name + ":")
    print(test)


phase_separating_poi_id = 25

itemSimilarityRecommender = ItemSimilarityRecommender()
plainUserRecommender = PlainUserRecommender()
usercontextBasedRecommender = UserContextBasedRecommender()
# usercontextBasedRecommender.additional_preference_training(1)
# recommended_pois = usercontextBasedRecommender.context_mf(1)
# print(recommended_pois)
# usercontextBasedRecommender.additional_preference_training(1)
# recommended_pois = usercontextBasedRecommender.context_mf(1)
# print(recommended_pois)
# recommended_pois = plainUserRecommender.context_mf(4, poi_start_id=25)

if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
