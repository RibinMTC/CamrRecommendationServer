import json
import pathlib

import boto3
from boto3.dynamodb.conditions import Key


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

    def query_table(self, table_name, attribute_name, attribute_value):
        table = self.dynamodb.Table(table_name)

        response = table.query(
            KeyConditionExpression=Key(attribute_name).eq(attribute_value)
        )

        return response['Items']

    def load_all_users(self):
        return self.scan_table('camr-users')

    def load_all_pois(self):
        return self.scan_table('camr-poi-storage')

    def load_all_google_user_prefs(self):
        return self.scan_table('camr-user-item-preferences')

    def load_all_real_user_prefs(self):
        return self.scan_table('camr-real-user-item-preferences')

    def load_all_real_user_likes(self, user_id):
        return self.query_table('camr-user-item-likes-temp', 'UserId', user_id)


if __name__ == '__main__':
    aws = AwsManager()

    # users = aws.load_all_users()
    #
    # pois = aws.load_all_pois()
    #
    # user_prefs = aws.load_all_google_user_prefs()

    #test = aws.load_all_real_user_likes(1)
    #print(test)
    # Uncomment code below to iterate through all pois and its attributes

    # for poi in pois:
    #     print("Poi Id: " + str(poi['PoiId']))
    #     print("Poi Name: " + str(poi['Name']))
    #     print("Poi Categories: " + str(poi['Categories']))
    #     print("Rating: " + str(poi['Rating']))
    #     print("Reviews: " + str(poi['Reviews']))

    # Uncomment code below to iterate through all users and its attributes
    # for user in users:
    #     print("User Id: " + str(user['UserId']))
    #     print("Age: " + str(user['Age']))
    #     print("Gender: " + str(user['Gender']))
    #     print("Name: " + str(user['Name']))
    #     print("FieldsOfInterest: " + str(user['FieldsOfInterest']))
    #     print("PersonalityTraits: " + str(user['PersonalityTraits']))

    # Uncomment code below to iterate through all user preferences
    # for user_pref in user_prefs:
    #     print("UserId: " + str(user_pref['UserId']))
    #     print("PoiId: " + str(user_pref['PoiId']))
    #     print("Preference: " + str(user_pref['UserPreference']))


