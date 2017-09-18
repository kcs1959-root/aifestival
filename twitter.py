import tweepy
import pprint
from PIL import Image
import numpy as np
import urllib.request
import io
from model import Classifier

pp = pprint.PrettyPrinter(width=41, compact=True)
classifier = Classifier()

# credentials.py
from credentials import *
auth = tweepy.OAuthHandler(CONSUMER_TOKEN, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)
        if status.in_reply_to_screen_name == api.me().screen_name:
            pp.pprint(status)

            if 'media' in status.entities:
                url = status.entities['media'][0]['media_url']
                with urllib.request.urlopen(url) as u:
                    f = io.BytesIO(u.read())
                image = Image.open(f)
                # image = image.thumbnail((28, 28))
                image = image.convert('L')
                image = 1.0 - np.asarray(image, dtype="float32") / 255
                image = image.reshape((1,784))

                prediction = classifier.predict(image)
                pp.pprint(prediction)
                # api.update_status(jsonify({ str(k): float(v * 100) for k,v in prediction.items() }))



myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
# myStream.userstream(_with="user")
myStream.userstream()
# public_tweets = api.home_timeline()
# for tweet in public_tweets:
#     print(tweet.text)
