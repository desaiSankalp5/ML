from flask import Flask, render_template
import twitter_data
import get_stock_data
import pandas as pd
from pandas import DataFrame
from datetime import date
import csv
from StringIO import StringIO
from zipfile import ZipFile
from urllib import urlopen
import re
import numpy as np
from sklearn.model_selection import KFold
import datetime
import pygal
from nsetools import Nse
from sklearn import svm, grid_search
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from dateutil import parser
from nsepy import get_history
import datetime
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#from matplotlib import style
#import matplotlib.pyplot as plt,mpld3
#import matplotlib.animation as animation
#import json

app = Flask(__name__)

# Global Vars

tweet_s = []
afinn = dict()
csvFile = open('Data/Tweets.csv', 'w')
csvWriter = csv.writer(csvFile)
startDate = datetime.date(2020,2,18)
endDate = datetime.date(2020,5,26) 



# Set to False when deploying
app.debug = True

@app.route('/')
def welcome():
	nse = Nse()
	top_gainers = nse.get_top_gainers()
	top_0 = top_gainers[0]
	top_0_symbol = top_0['symbol']
	top_0_price = top_0['previousPrice']
	
	top_1 = top_gainers[1]
	top_1_symbol = top_1['symbol']
	top_1_price = top_1['previousPrice']
	
	top_2 = top_gainers[2]
	top_2_symbol = top_2['symbol']
	top_2_price = top_2['previousPrice']
	
	top_3 = top_gainers[3]
	top_3_symbol = top_3['symbol']
	top_3_price = top_3['previousPrice']
	
	top_4 = top_gainers[4]
	top_4_symbol = top_4['symbol']
	top_4_price = top_4['previousPrice']
	
	top_losers = nse.get_top_losers()
	top_5 = top_losers[0]
	top_5_symbol = top_5['symbol']
	top_5_price = top_5['previousPrice']
	
	top_6 = top_losers[1]
	top_6_symbol = top_6['symbol']
	top_6_price = top_6['previousPrice']
	
	top_7 = top_losers[2]
	top_7_symbol = top_7['symbol']
	top_7_price = top_7['previousPrice']
	
	top_8 = top_losers[3]
	top_8_symbol = top_8['symbol']
	top_8_price = top_8['previousPrice']
	
	top_9 = top_losers[4]
	top_9_symbol = top_9['symbol']
	top_9_price = top_9['previousPrice']
	
	
	
	
	infy = nse.get_quote('infy')
	infy1 = infy['lastPrice']
	infyclosing = infy['previousClose']
	diffInfy = infy1 - infyclosing
	if(diffInfy > 0):
		icolor = 'green'
		iarrow = 'up'
	else:
		icolor = 'red'
		iarrow = 'bottom'
	infyans = manipulate(infy1,infyclosing)
	
	tcs = nse.get_quote('tcs')
	tcs1 = tcs['lastPrice']
	tcsclosing = tcs['previousClose']
	diffTcs = tcs1 - tcsclosing
	if(diffTcs > 0):
		tcolor = 'green'
		tarrow = 'up'
	else:
		tcolor = 'red'
		tarrow = 'bottom'
	tcsans = manipulate(tcs1,tcsclosing)

	itc = nse.get_quote('itc')
	itc1 = itc['lastPrice']
	itcclosing = itc['previousClose']
	diffItc = itc1 - itcclosing
	if(diffItc > 0):
		itccolor = 'green'
		itcarrow = 'up'
	else:
		itccolor = 'red'
		itcarrow = 'bottom'
	
	itcans = manipulate(itc1,itcclosing)

	
	ioc = nse.get_quote('ioc')
	ioc1 = ioc['lastPrice']
	iocclosing = ioc['previousClose']
	diffioc = ioc1 - iocclosing
	if(diffioc > 0):
		iocolor = 'green'
		ioarrow = 'up'
	else:
		iocolor = 'red'
		ioarrow = 'bottom'
	iocans = manipulate(ioc1,iocclosing)
	
	
	
	
	return render_template('main.html',infy1=infy1,infy2=infyans,
							tcs1=tcs1,tcs2=tcsans,
							itc1=itc1,itc2 = itcans,
							ioc1=ioc1,ioc2 = iocans,
							symbol0 = top_0_symbol,
							price0=top_0_price,
							symbol1=top_1_symbol,
							price1=top_1_price,
							symbol2=top_2_symbol,
							price2=top_2_price,
							symbol3=top_3_symbol,
							price3=top_3_price,
							symbol4=top_4_symbol,
							price4=top_4_price,
							symbol5=top_5_symbol,
							price5=top_5_price,
							symbol6=top_6_symbol,
							price6=top_6_price,
							symbol7=top_7_symbol,
							price7=top_7_price,
							symbol8=top_8_symbol,
							price8=top_8_price,
							symbol9=top_9_symbol,
							price9=top_9_price,icolor=icolor,tcolor=tcolor,itcarrow=itcarrow,itccolor=itccolor,iocolor=iocolor,
							ioarrow=ioarrow)
	
	
@app.route('/REL')	
def rel():
	keyword = '$RELIANCE'
	stockSymbol = 'REL'
	my_result = fetch(keyword,stockSymbol)
	return render_template('index.html', t=my_result)

@app.route('/INFY')
def infy():
	keyword = '$INFY'
	stockSymbol = 'INFY'
	my_result = fetch(keyword,stockSymbol)
	
	predicted_value = my_result[0]
	trend = my_result[1]
	
	
	if(trend == 'Downtrend'):
		sentimentColor = 'red'
		sentimentArrow = 'bottom'
	else:
		sentimentColor = 'green'
		sentimentArrow = 'up'
	
	quarterlyRepResult = 'By analysing reports of previous 4 quarters, it can be observed that the Net Profit After Tax was on a high upto December 2017. \
						But, the lastest quarter revealed that the value has decreased gradually and the current company performance is not optimal. '
	
	
	nse = Nse()
	q = nse.get_quote('infy')
	q1 = q['closePrice']
	infy = pd.read_csv('E:\College\Final Year Project\Dataset\INFY.NS.csv')
	contents = ['Date','Open','High','Low','Close','Adj Close','Volume']
	infy.columns = contents
	infy.drop('Adj Close',axis=1 ,inplace = True)
	
	tempOpen = infy['Open']
	openList = tempOpen.tolist()
	tempClose = infy['Close']
	closeList = tempClose.tolist()
	tempDate = infy['Date']
	dateList0 = tempDate.tolist()
	dateList = dateList0.reverse()
	
	trendValue = float(predicted_value) - q1
	if(trendValue > 0):
		color = 'green'
		arrow = 'top'
	if(trendValue < 0):
		color = 'red'	
		arrow = 'bottom'
	
	
	
	

	data = pd.ExcelFile('infosys1.xlsx')
	data2 = data.parse(header=1,parse_cols='A:E')
	data2.columns = ['Description','Mar-18','Dec-17','Sept-17','June-17']
	cleaned_wdata = data2[data2['Description'].notnull()]
	cleaned_wdata2 = cleaned_wdata[cleaned_wdata['Mar-18'].notnull()]
	cleaned_wdata2.set_index(['Description'], inplace=True)


	try:	
		graph = pygal.Line()
		graph.title = 'Infosys Timeline'
		
		graph.x_labels = dateList0
		graph.add('Open', openList)
		graph.add('Close', closeList)
		graph_data = graph.render_data_uri()
		return render_template('infosys.html', t=[cleaned_wdata2.to_html(classes='female')], 
								q=q1,  
								graph_data = graph_data,
								predicted_value=predicted_value,
								trend=trend,
								color=color,arrow=arrow,sentimentColor=sentimentColor,sentimentArrow=sentimentArrow,quarterlyRepResult=quarterlyRepResult)
	
	except Exception, e:
		return(str(e))
	
	
@app.route('/TCS')
def tcs():
	keyword = '$TCS'
	stockSymbol = 'TCS'
	my_result = fetch(keyword,stockSymbol)
	
	predicted_value = my_result[0]
	trend = my_result[1]
	if(trend == 'Downtrend'):
		sentimentColor = 'red'
		sentimentArrow = 'bottom'
	else:
		sentimentColor = 'green'
		sentimentArrow = 'up'
	
	nse = Nse()
	q = nse.get_quote('tcs')
	q1 = q['closePrice']
	tcs = pd.read_csv('E:\College\Final Year Project\Dataset\TCS.NS.csv')
	contents = ['Date','Open','High','Low','Close','Adj Close','Volume']
	tcs.columns = contents
	tcs.drop('Adj Close',axis=1 ,inplace = True)
	
	tempOpen = tcs['Open']
	openList = tempOpen.tolist()
	tempClose = tcs['Close']
	closeList = tempClose.tolist()
	tempDate = tcs['Date']
	dateList0 = tempDate.tolist()
	dateList = dateList0.reverse()
	
	trendValue = float(predicted_value) - q1
	if(trendValue > 0):
		color = 'green'
		arrow = 'top'
	if(trendValue < 0):
		color = 'red'	
		arrow = 'bottom'
	
	try:	
		graph = pygal.Line()
		graph.title = 'Tata Consultancy Services Timeline'
		
		graph.x_labels = dateList0
		graph.add('Open', openList)
		graph.add('Close', closeList)
		graph_data = graph.render_data_uri()
		return render_template('tcs.html',  
								q=q1,  
								graph_data = graph_data,
								predicted_value=predicted_value,
								trend=trend,
								color=color,arrow=arrow,sentimentColor = sentimentColor,sentimentArrow=sentimentArrow)
	
	except Exception, e:
		return(str(e))
	
	

@app.route('/HDFCBANK')
def hdfcbank():
	keyword = '$HDFCBANK'
	stockSymbol = 'HDFCBANK'
	my_result = fetch(keyword,stockSymbol)
	return render_template('index.html', t=my_result)

@app.route('/ITC')
def itc():
	keyword = '$ITC'
	stockSymbol = 'ITC'
	
	my_result = fetch(keyword,stockSymbol)
	
	predicted_value = my_result[0]
	trend = my_result[1]
	
	if(trend == 'Downtrend'):
		sentimentColor = 'red'
		sentimentArrow = 'bottom'
	else:
		sentimentColor = 'green'
		sentimentArrow = 'up'
	
	nse = Nse()
	q = nse.get_quote('itc')
	q1 = q['closePrice']
	itc = pd.read_csv('E:\College\Final Year Project\Dataset\ITC.NS.csv')
	contents = ['Date','Open','High','Low','Close','Adj Close','Volume']
	itc.columns = contents
	itc.drop('Adj Close',axis=1 ,inplace = True)
	
	tempOpen = itc['Open']
	openList = tempOpen.tolist()
	tempClose = itc['Close']
	closeList = tempClose.tolist()
	tempDate = itc['Date']
	dateList0 = tempDate.tolist()
	dateList = dateList0.reverse()
	
	trendValue = float(predicted_value) - q1
	if(trendValue > 0):
		color = 'green'
		arrow = 'top'
	if(trendValue < 0):
		color = 'red'	
		arrow = 'bottom'
	
	try:	
		graph = pygal.Line()
		graph.title = 'ITC Timeline'
		
		graph.x_labels = dateList0
		graph.add('Open', openList)
		graph.add('Close', closeList)
		graph_data = graph.render_data_uri()
		return render_template('itc.html',  
								q=q1,  
								graph_data = graph_data,
								predicted_value=predicted_value,
								trend=trend,sentimentColor=sentimentColor,sentimentArrow=sentimentArrow)
	
	except Exception, e:
		return(str(e))
	
	

@app.route('/HINDUNILVR')
def hindunilvr():
	keyword = '$HINDUNILVR'
	my_result = fetch(keyword)
	return render_template('index.html', t=my_result)	
	
@app.route('/LTI')
def lti():
	keyword = '$LTI'
	my_result = fetch(keyword)
	return render_template('index.html', t=my_result)	

@app.route('/WIPRO')
def wipro():
	keyword = '$WIPRO'
	my_result = fetch(keyword)
	return render_template('index.html', t=my_result)	
	

	
@app.route('/IOC')
def ioc():
	keyword = '$IOC'
	stockSymbol = 'IOC'
	my_result = fetch(keyword,stockSymbol)
	
	predicted_value = my_result[0]
	trend = my_result[1]
	
	if(trend == 'Downtrend'):
		sentimentColor = 'red'
		sentimentArrow = 'bottom'
	else:
		sentimentColor = 'green'
		sentimentArrow = 'up'
	
	nse = Nse()
	q = nse.get_quote('ioc')
	q1 = q['closePrice']
	ioc = pd.read_csv('E:\College\Final Year Project\Dataset\IOC.NS.csv')
	contents = ['Date','Open','High','Low','Close','Adj Close','Volume']
	ioc.columns = contents
	ioc.drop('Adj Close',axis=1 ,inplace = True)
	
	tempOpen = ioc['Open']
	openList = tempOpen.tolist()
	tempClose = ioc['Close']
	closeList = tempClose.tolist()
	tempDate = ioc['Date']
	dateList0 = tempDate.tolist()
	dateList = dateList0.reverse()
	
	trendValue = float(predicted_value) - q1
	if(trendValue > 0):
		color = 'green'
		arrow = 'top'
	if(trendValue < 0):
		color = 'red'	
		arrow = 'bottom'
	
	try:	
		graph = pygal.Line()
		graph.title = 'IOC Timeline'
		
		graph.x_labels = dateList0
		graph.add('Open', openList)
		graph.add('Close', closeList)
		graph_data = graph.render_data_uri()
		return render_template('ioc.html',  
								q=q1,  
								graph_data = graph_data,
								predicted_value=predicted_value,
								trend=trend,
								color=color,arrow=arrow,sentimentColor=sentimentColor,sentimentArrow=sentimentArrow)
	
	except Exception, e:
		return(str(e))
	
@app.route('/MARUTI')
def maruti():
	keyword = '$MARUTI'
	my_result = fetch(keyword)
	return render_template('index.html', t=my_result)	
	
@app.route('/ICICIBANK')	
def icicibank():
	keyword = '$ICICIBANK'
	my_result = fetch(keyword)
	return render_template('index.html', t=my_result)
	
def manipulate(current,closing):
	diff = current-closing
	ratio = (float)(diff/closing)*100
	if(diff > 0):
		ans = "+"+str(diff)+" ("+str(round(ratio,2))+"%)"
	else :
		ans = str(diff)+" ("+str(round(ratio,2))+"%)"

	return ans
	
def fetch(keyword,stockSymbol):
	twitterData = twitter_data.TwitterData('2020-2-18')
	tweets = twitterData.getTwitterData(keyword)
	#Twitter data fetched
	
	keyword2 = keyword
	
	historical_data = get_history(symbol=stockSymbol,start= datetime.date(2020,2,18),end = datetime.date(2020,5,26))
	data = pd.DataFrame(historical_data, columns=['Open', 'Close', 'High', 'Low'])
	data.reset_index(level=0, inplace=True)
	#print(data)
	
	open_price = {}
	close_price = {}
	high_price = {}
	low_price = {}

	for index, row in data.iterrows():
		date = row['Date']
		open_price.update({date: row['Open']})
		close_price.update({date: row['Close']})
		high_price.update({date: row['High']})
		low_price.update({date: row['Low']})
	#Stock data fetched
	
	for t in tweets.items():
		for value in t[1]:
			tweet_s.append(value)
	
	url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
	zipfile = ZipFile(StringIO(url.read()))
	afinn_file = zipfile.open('AFINN/AFINN-111.txt')

	
	for line in afinn_file:
		parts = line.strip().split()
		if len(parts) == 2:
			afinn[parts[0]] = int(parts[1])

	sentiment_analyzer()
	inputTweets = csv.reader(open('Data/Tweets.csv', 'rb'), delimiter=',')
	stopWords = getListOfStopWords()
	count = 0;
	featureList = []
	list_tweet = []
	labelList = []
	dates = []
	tweets = []
	date_split = []

	for row in inputTweets:
		#print(row)
		if len(row) == 4:
			list_tweet.append(row)
			sentiment = row[0]
			date = row[1]
			text = row[2]
			
			date_split.append(date)
			dates.append(date)
			labelList.append(sentiment)
			
			
			processedText = processTweetText(text)
			featureVector = getFeatureVector(processedText, stopWords)
			featureList.extend(featureVector)
			tweets.append((featureVector, sentiment))
			#print(tweets)
	#print(featureList)

	result = getFeatureVectorAndLabels(tweets, featureList)
	#return result
	#new code
	
	
	data2 = open('newfile.txt', 'r')
	files = np.loadtxt(data2,dtype=str, delimiter=',')

	#Now we will split the data
	inp_data2 = []
	inp_data2 = np.array(files[:,0:-1], dtype='float')
	target2 = np.array(files[:,-1],dtype='int')

	X = np.array(inp_data2)
	y = np.array(target2)
	best_params_ = svc_param_selection(X,y,6)
	svc_RBF = svm.SVC(kernel='rbf', C=10,gamma=0.01).fit(X, y)
	#print("accuracy of RBF Kernel with gamma=0.01 is ", svc_RBF.score(X,y))
	#return svc_RBF.score(X,y)
	#accuracy score 
	#checkpoint 2
	
	#print "Preparing dataset for stock prediction using stock data and tweet sentiment...."
	date_tweet_details = {}
	file = open("stockpredict.txt", "w")
	totalPositiveCount=0
	totalNeutralCount=0
	totalNegativeCount=0
	myList = []
	final = []
	for dateVal in np.unique(date_split):
		date_totalCount = 0
		date_PosCount = 0
		date_NegCount = 0
		date_NutCount = 0
		total_sentiment_score = 0
		
			
		for row in list_tweet:
			sentiment = row[0]
			temp_date = row[1]
			sentiment_score = row[3]
			if(temp_date == dateVal):
				total_sentiment_score += float(sentiment_score)
				date_totalCount+=1
				if (sentiment == 'positive'):
					date_PosCount+=1
				elif (sentiment == 'negative'):
					date_NegCount+=1
				elif (sentiment == 'neutral'):
					date_NutCount+=1
				   
		s = str(date_totalCount)+" "+str(date_PosCount)+" "+str(date_NegCount)+" "+str(date_NutCount)
		date_tweet_details.update({dateVal: s})
		
		totalPositiveCount += date_PosCount
		totalNeutralCount += date_NegCount
		totalNegativeCount += date_NutCount
		

		
	   
		dateVal = dateVal.strip()
		day = datetime.datetime.strptime(dateVal, '%Y-%m-%d').strftime('%A')
		#print dateVal
		#print day
		closing_price = 0.
		opening_price = 0.
		if day == 'Saturday':
			update_date = dateVal.split("-")
			if len(str((int(update_date[2])-1)))==1:
				dateVal = update_date[0]+"-"+update_date[1]+"-0"+str((int(update_date[2])-1))
			else:
				dateVal = update_date[0] + "-" + update_date[1] + "-" + str((int(update_date[2]) - 1))
		   
			dt = parser.parse(dateVal)
			datetime_obj = dt.date()
			opening_price = open_price[datetime_obj]
			closing_price = close_price[datetime_obj]
		elif day == 'Sunday':
			update_date = dateVal.split("-")
			if len(str((int(update_date[2])-2)))==1:
				dateVal = update_date[0]+"-"+update_date[1]+"-0"+str((int(update_date[2])-2))
			else:
				dateVal = update_date[0] + "-" + update_date[1] + "-" + str((int(update_date[2]) - 2))
			   
			dt = parser.parse(dateVal)
			datetime_obj = dt.date()
			opening_price = open_price[datetime_obj]
			closing_price = close_price[datetime_obj]
		else:
			dt = parser.parse(dateVal)
			datetime_obj = dt.date()
			opening_price = open_price[datetime_obj]
			closing_price = close_price[datetime_obj]
	   
	   
	   
		#print dateVal
		#print "Total tweets = ", date_totalCount, " Positive tweets = ", date_PosCount, " Negative tweets = ", date_NegCount
		#print "Total sentiment score = ", total_sentiment_score
		market_status = 0
		if (float(closing_price)-float(opening_price)) > 0:
			market_status = 1
		else:
			market_status =-1
		file.write( str(date_PosCount) + "," + str(date_NegCount) + "," + str(date_NutCount) +"," + str(date_totalCount) + "," + str(market_status) + "\n")
		#print " Total Tweet For date =",dateVal ," Count =" , date_totalCount
		#print " Positive Tweet For date =",dateVal ," Count =" , date_PosCount
		#print " Negative Tweet For date =",dateVal ," Count =" , date_NegCount
		#print " Neutral Tweet For date =",dateVal ," Count =" , date_NutCount
	file.close()
	#print "Read from text file and prepare data matrix & target matrix...."
	
	data_Stock = open('stockpredict.txt', 'r')
	inp_dataStock = []
	stockfiles = np.loadtxt(data_Stock, delimiter=',')
	inp_dataStock = np.array(stockfiles[:,0:-1],dtype = 'float')
	stock_Y = stockfiles[:,-1]

	X_stock = np.array(inp_dataStock)
	y_stock = np.array(stock_Y)
	#best_params_1 = svc_param_selection_1(X_stock,stock_Y,4)
	
	svc_RBF = svm.SVC(kernel='rbf', C=10,gamma=0.01).fit(X_stock, y_stock)
	
	#print("accuracy of RBF Kernel with gamma=0.01 is ", svc_RBF.score(X,y))
	#return svc_RBF.score(X_stock,y_stock)
	#checkpoint3
	dates = []
	prices = []

	for index, row in data.iterrows():
	   date = row['Date']
	   int_date = date.strftime('%Y%m%d')
	   #dates.append(int(int_date.split('-')[2]))
	   dates.append(int_date)
	   prices.append(float(row['Close']))

	#print dates
	#print prices
	predicted_price = predict_price(dates, prices,20200526)	
	bhavishya = str(predicted_price)
	
	final.append(bhavishya)
	
	#print "\nThe stock close price for 25th May will be:"
	#print "RBF kernel: Rs.", str(predicted_price[0])'''
	myList.append([totalPositiveCount,totalNegativeCount,totalNeutralCount])
	#maximum = myList.index(max(myList))
	
	
	
	if(totalPositiveCount > totalNegativeCount):
		if(totalPositiveCount > totalNeutralCount):
			final.append('Uptrend')
			return final
	
	if(totalNegativeCount > totalPositiveCount):
		if(totalNegativeCount > totalNeutralCount):
			final.append('Downtrend')
			return final
			
	else :
		final.append('Neutral')
		return final
	
	
def predict_price(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
	svr_rbf.fit(dates, prices) # fitting the data points in the models
	return svr_rbf.predict(x)[0]
	
	
def svc_param_selection(X, y,nfolds):
	Cs = [0.001, 0.01, 0.1, 1, 10]
	gammas = [0.001, 0.01, 0.1, 1]
	param_grid = {'C': Cs, 'gamma' : gammas}
	grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
	grid_search.fit(X, y)
	grid_search.best_params_
	return grid_search.best_params_
	

	
	
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
    #return {'feature_vector' : feature_vector, 'labels': labels}
	#new code starts from here
	
	
	
	
	
if __name__ == "__main__":
    app.run()	