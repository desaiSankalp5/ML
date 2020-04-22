import datetime
from datetime import timedelta
import pickle
import json
import os
import oauth2
import urllib

class TwitterData :

    def __init__(self,currentDate):
        self.currentDate = datetime.datetime.strptime(currentDate,'%Y-%m-%d')
        #print(type(self.currentDate))
        #print(self.currentDate)
        self.dates = []
        self.dates.append(self.currentDate.strftime("%Y-%m-%d"))
        for i in range(1,10):
            difference = timedelta(days=-i)
            newDate = self.currentDate + difference
            self.dates.append(newDate.strftime("%Y-%m-%d"))
        #print(self.dates)

    def getTwitterData(self, keyword):
        self.weekTweets = {}
        for i in range(0, 9):
            params = {'since': self.dates[i + 1], 'until': self.dates[i]}
            self.weekTweets[i] = self.getData(keyword, params)

        # Write data to a pickle file
        filename = 'Data/weekTweets_' + keyword + '_' + datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S") + '.txt'
        outfile = open(filename, 'wb')
        pickle.dump(self.weekTweets, outfile)
        outfile.close()
        return self.weekTweets

    def parse_config(self):
      config = {}
      if os.path.exists('config.json'):
          with open('config.json') as fp:
              config.update(json.load(fp))
      else:
          print("Config file missing")
      return config

    def oauth_req(self, url, http_method="GET", post_body=None,http_headers=None):
      config = self.parse_config()
      consumer = oauth2.Consumer(key=config.get('consumer_key'), secret=config.get('consumer_secret'))
      token = oauth2.Token(key=config.get('access_token'), secret=config.get('access_token_secret'))
      client = oauth2.Client(consumer, token)

      resp, content = client.request(
          url,
          method=http_method,
          body=post_body or '',
          headers=http_headers
      )
      return content

    def getData(self, keyword, params = {}):
        maxTweets = 200
        url = 'https://api.twitter.com/1.1/search/tweets.json?'
        data = {'q': keyword, 'lang': 'en', 'result_type': 'mixed', 'since_id': 2014,'count': maxTweets, 'include_entities': 0}

        # Add if additional params are passed
        if params:
            for key, value in params.iteritems():
                data[key] = value

        #print(data)

        url += urllib.urlencode(data)
        #print(url)

        response = self.oauth_req(url)
        jsonData = json.loads(response)
        tweets = []
        if 'errors' in jsonData:
            print ("API Error")
            print (jsonData['errors'])
        else:
            for item in jsonData['statuses']:
                d = datetime.datetime.strptime(item['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                str = d.strftime('%Y-%m-%d')+" | "+item['text'].replace('\n', ' ')
                tweets.append(str)
        return tweets

'''if __name__ == "__main__":
    s = TwitterData('2017-11-05')
    tweets = s.getTwitterData('$AAPL')
    tweet_s = []
    for s in tweets.items():
        for value in s[1]:
            tweet_s.append(value)

    print(tweet_s)'''


