
# coding: utf-8

# In[ ]:


import twitter_data
import pandas as pd
from datetime import date
import csv
from StringIO import StringIO
from zipfile import ZipFile
from urllib import urlopen
import re
import numpy as np


# In[2]:


symbol = raw_input("Please Enter Keyword of any NSE stock : ")

while not symbol:
    symbol = raw_input("Please enter Keyword: ")

keyword = '$'+symbol


# In[3]:


print("Fetching Twitter Data for "+keyword+" Company")

twitterData = twitter_data.TwitterData('2020-2-18')
tweets = twitterData.getTwitterData(keyword)

print("Fetched Twitter Data Successfully !!")


# In[4]:


tweet_s = []
for t in tweets.items():
    for value in t[1]:
        tweet_s.append(value)

csvFile = open('Data/Tweets.csv', 'w')
csvWriter = csv.writer(csvFile)

url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
zipfile = ZipFile(StringIO(url.read()))
afinn_file = zipfile.open('AFINN/AFINN-111.txt')

afinn = dict()
for line in afinn_file:
    parts = line.strip().split()
    if len(parts) == 2:
        afinn[parts[0]] = int(parts[1])


def tokenize(text):
    return re.sub('\W+', ' ', text.lower()).split()

def afinn_sentiment(terms, afinn):
    total = 0.
    for t in terms:
        if t in afinn:
            total += afinn[t]
    return total

def sentiment_analyzer():
    tokens = [tokenize(t) for t in tweet_s]

    #tokens_2 = [nltk.word_tokenize(t) for t in tweet_s]

    afinn_total = []
    for tweet in tokens:
        total = afinn_sentiment(tweet, afinn)
        afinn_total.append(total)

    positive_tweet_counter = []
    negative_tweet_counter = []
    neutral_tweet_counter = []
    for i in range(len(afinn_total)):
        if afinn_total[i] > 0:
            positive_tweet_counter.append(afinn_total[i])
            csvWriter.writerow(["positive", tweet_s[i].encode('utf-8').split("|")[0], tweet_s[i].encode('utf-8').split("|")[1], afinn_total[i]])
        elif afinn_total[i] < 0:
            negative_tweet_counter.append(afinn_total[i])
            csvWriter.writerow(["negative", tweet_s[i].encode('utf-8').split("|")[0], tweet_s[i].encode('utf-8').split("|")[1], afinn_total[i]])
        else:
            neutral_tweet_counter.append(afinn_total[i])
            csvWriter.writerow(["neutral", tweet_s[i].encode('utf-8').split("|")[0], tweet_s[i].encode('utf-8').split("|")[1], afinn_total[i]])

print("Processing Tweets")
sentiment_analyzer()

def getListOfStopWords():
    stopWords = []
    fp = open('Data/stopwords.txt', 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords

def processTweetText(text):
    text = text.lower()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text)
    text = re.sub('@[^\s]+','AT_USER',text)
    text = re.sub('[\s]+', ' ', text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    text = text.strip('\'"')
    return text

def getFeatureVector(text, stopWords):
    featureVector = []
    words = text.split()
    for w in words:
        w = w.strip('\'"?,.')
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector

# start replaceTwoOrMore
def replaceTwoOrMore(s):
    # look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
# end

def getFeatureVectorAndLabels(tweets, featureList):
    #print(featureList)
    sortedFeatures = sorted(featureList)
    feature_vector = []
    labels = []
    file = open("newfile.txt", "w")

    for t in tweets:
        label = 0
        map = {}

        tweet_words = t[0]
        tweet_opinion = t[1]

        for w in sortedFeatures:
            map[w] = 0

        # Fill the map
        for word in tweet_words:
            word = replaceTwoOrMore(word)
            word = word.strip('\'"?,.')
            if word in map:
                map[word] = 1
        # end for loop
        values = map.values()
        feature_vector.append(values)
        if (tweet_opinion == 'positive'):
            label = 0
        elif (tweet_opinion == 'negative'):
            label = 1
        elif (tweet_opinion == 'neutral'):
            label = 2
        labels.append(label)
        feature_vector_value = str(values).strip('[]')
        file.write(feature_vector_value + "," + str(label) + "\n")
    file.close()
    return {'feature_vector' : feature_vector, 'labels': labels}
#end


inputTweets = csv.reader(open('Data/Tweets.csv', 'rb'), delimiter=',')
stopWords = getListOfStopWords()
count = 0;
featureList = []
tweets = []
print "Creating feature set and generating feature matrix...."

for row in inputTweets:
    print(row)
    if len(row) == 4:
        sentiment = row[0]
        date = row[1]
        text = row[2]
        processedText = processTweetText(text)
        featureVector = getFeatureVector(processedText, stopWords)
        featureList.extend(featureVector)
        tweets.append((featureVector, sentiment))
        #print(tweets)
#print(featureList)

result = getFeatureVectorAndLabels(tweets, featureList)
print "Dataset is ready \n"

