# import main Flask class and request object
from flask import Flask, request, jsonify

# create the Flask app
from src.Item_Similarity_Recommender import ItemSimilarityRecommender
from src.Restaurant_Recommender import get_popular_items
from src.usercontext_database_creator import userattributescodes
from src.usercontext_recommendation import UserContextRecommender

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

        last_seen_poi = request_data['poiNames'][0]
        similar_pois = itemSimilarityRecommender.get_similar_items(last_seen_poi).tolist()
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
        recommended_pois = userContextRecommender.context_mf(user_id)
        recommendation["poiNames"] = recommended_pois

        return jsonify(recommendation)

    except Exception as e:
        print(e)
        return recommendation


def get_similar_items_test(poi_name):
    # test = get_popular_items( ["Bierhalle Wolf", "Rheinfelder Bierhalle", "Chop-Stick Restaurant", "Walliser Keller
    # \"The SwissRestaurant\""]) print(test)

    test = itemSimilarityRecommender.get_similar_items(poi_name)
    print("Similar restaurants to " + poi_name + ":")
    print(test)


if __name__ == '__main__':
    # run app in debug mode on port 5000

    itemSimilarityRecommender = ItemSimilarityRecommender()
    userContextRecommender = UserContextRecommender(len(userattributescodes))
    # recommended_pois = userContextRecommender.context_mf(1)
    # print(recommended_pois)
    # recommended_pois = userContextRecommender.context_mf(1)
    # print(recommended_pois)
    # get_similar_items_test("Rheinfelder Bierhalle")
    app.run(port=5000, host='0.0.0.0')
    # serve(app, host='0.0.0.0', port=5000, threads=1)

