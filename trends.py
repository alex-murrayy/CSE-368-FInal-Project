import yfinance as yf
import pandas as pd
import numpy as np
from stock_features import StockFeatures
from sklearn.metrics import precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Ask user for ticker symbol
ticker = input("Enter the ticker symbol: ").strip().upper()

stock = yf.Ticker(ticker)
stock_data = stock.history(period="max")

# del stock_data["Dividends"]
# del stock_data["Stock Splits"]

stock_data.plot.line(y="Close", use_index=True)

stock_data["Tomorrow"] = stock_data["Close"].shift(-1)
stock_data["Target"] = (stock_data["Tomorrow"] > stock_data["Close"]).astype(int)

# Predicting the movement of the stock
# Filter data from 1990 onwards
stock_data = stock_data.loc["1975-01-01":].copy()

# Add features using StockFeatures class
features = StockFeatures()
stock_data = features.features(stock_data)


# Drop the last NaN row in Tomorrow 
stock_data = stock_data.dropna()

#--- AI Generated Code ---#
# Diagnostic: Check for NaN values in Target
print("NaN in Target:", stock_data["Target"].isna().sum())
print("\nTarget distribution (before train/test split):")
print(stock_data["Target"].value_counts())
print(f"Percentage of 1s (price goes up): {stock_data['Target'].mean()*100:.2f}%")
#--- AI Generated Code ---#

# N estimators is the number of trees in the forest
# min_samples_split is the minimum number of samples required to split an internal node
# random_state is the random seed for reproducibility
model = RandomForestClassifier(n_estimators=400, min_samples_split=50, max_depth=15, random_state=1, n_jobs=-1)

train = stock_data.iloc[:-100]
test = stock_data.iloc[-100:]

#--- AI Generated Code ---#
# Diagnostic: Check class distribution in train and test sets
print("\n" + "="*50)
print("TRAINING SET:")
print("Target distribution:", train["Target"].value_counts())
print(f"Percentage of 1s: {train['Target'].mean()*100:.2f}%")
print(f"Training samples: {len(train)}")

print("\nTEST SET:")
print("Target distribution:", test["Target"].value_counts())
print(f"Percentage of 1s: {test['Target'].mean()*100:.2f}%")
print(f"Test samples: {len(test)}")
#--- AI Generated Code ---#

# Get predictor names 
predictor_names = features.predictors()
# get predictors that are in the data
predictors = features.validate(stock_data, predictor_names)

#--- AI Generated Code ---#
# Diagnostic: Check feature statistics
print("\n" + "="*50)
print("FEATURE STATISTICS (Training Set):")
print(train[predictors].describe())

print("\n" + "="*50)
print("Training model...")
#--- AI Generated Code ---#

model.fit(train[predictors], train["Target"])
preds = model.predict(test[predictors])

#--- AI Generated Code ---#
print("\n" + "="*50)
print("PREDICTIONS:")
print(f"Predictions array: {preds}")
print(f"Number of 0s predicted: {(preds == 0).sum()}")
print(f"Number of 1s predicted: {(preds == 1).sum()}")
#--- AI Generated Code ---#

# Convert predictions to pandas Series with same index as test data
preds_series = pd.Series(preds, index=test.index, name="Predictions")

# Calculate precision and accuracy
precision = precision_score(test["Target"], preds_series, zero_division=0)
accuracy = accuracy_score(test["Target"], preds_series)

#--- AI Generated Code ---#
print("\n" + "="*50)
print("MODEL PERFORMANCE METRICS:")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%) - Overall correctness of ALL predictions")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%) - Does not calculate true negatives")
print("\n" + "="*50)
#--- AI Generated Code ---#

