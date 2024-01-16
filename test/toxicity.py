# pip install google-api-python-client
# https://github.com/googleapis/google-api-python-client
from googleapiclient import discovery
from googleapiclient.errors import HttpError
# https://developers.perspectiveapi.com/s/docs-sample-requests?language=en_US
import time
import pandas as pd

GoogleCloud_API_KEY = "AIzaSyCGkpXa61kxYGaTi8fCkS67m2F_wvZF7_Q"

def get_toxicity_score(text):
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=GoogleCloud_API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    def get_toxic_by_txt(txt):
        if txt == "":
            return 0

        analyze_request = {
            'comment': {'text': txt},
            'requestedAttributes': {'TOXICITY': {}}
        }

        while True:
            try:
                response = client.comments().analyze(body=analyze_request).execute()
                time.sleep(30)
                break
            except Exception as err:
                print("TOXICITY FUNCTION: ", err)
                time.sleep(60)

        return response['attributeScores']['TOXICITY']['summaryScore']['value']


    if type(text) is str:
        return get_toxic_by_txt(text)
    elif type(text) is list:   # list
        df = pd.DataFrame()
        for i in range(len(text)):
            df.at[i, 'TOXICITY'] = get_toxic_by_txt(text[i])
        return df


if __name__ == "__main__":
    txt = "You are nasty."
    toxic_score = get_toxicity_score(txt)
    print("TEXT: {}".format(txt))
    print("SCORE: {}".format(toxic_score))