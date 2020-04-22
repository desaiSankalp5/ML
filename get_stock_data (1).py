from nsepy import get_history

class StockData:
	def __init__(self,startDate,endDate):
		self.startDate = startDate
		self.endDate = endDate

	def getStockData(self,keyword):
		historical_data = get_history(symbol=keyword, start=self.startDate, end=self.endDate)
		return historical_data

'''if __name__ == "__main__":
    response = raw_input("Please enter Keyword: ")
    while not response:
        response = raw_input("Please enter Keyword: ")
    print "Fetch stock finance data for "+response+" company "
    keyword = response
    stockData = StockData(date(2018,1,1),date(2018,3,5))
    historical_data = stockData.getStockData(keyword)
    data = pd.DataFrame(historical_data, columns=['Open', 'Close', 'High', 'Low'])
    data.reset_index(level=0, inplace=True)
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

    print "Real- Time Stock data fetched \n"'''
