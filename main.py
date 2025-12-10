from trends import StockTrendPredictor
from price import next_day_price

# Choose stock 
ticker = input("Enter the stock ticker: ")

# Predict tomorrow's trend
predictor = StockTrendPredictor(ticker)
predictor.predict()

print("PRICE PREDICTION:")
print("\n" + "="*50)

# Predict tomorrow's price
tomorrow_price = next_day_price(ticker)
print(f"Tomorrow's price: {tomorrow_price}")
