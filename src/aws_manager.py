import json
import pathlib

import boto3


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


if __name__ == '__main__':
    aws = AwsManager()

    users = aws.load_all_users()

    pois = aws.load_all_pois()

    user_prefs = aws.load_all_user_prefs()

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


