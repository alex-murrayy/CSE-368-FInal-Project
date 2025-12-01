import yfinance as yf
import pandas as pd
sp500 = yf.Ticker("^GSPC")
sp500_data = sp500.history(period="max")

sp500_data.plot.line(y="Close", use_index=True)

del sp500_data["Dividends"]
del sp500_data["Stock Splits"]

sp500_data["Tomorrow"] = sp500_data["Close"].shift(-1)
sp500_data["Target"] = (sp500_data["Tomorrow"] > sp500_data["Close"]).astype(int)

# Will stock go up or down?

# Filter data from 1990 onwards
sp500_data = sp500_data.loc["1990-01-01":].copy()

# Drop the last row which has NaN in Tomorrow (no future data available)
sp500_data = sp500_data.dropna()

# Diagnostic: Check for NaN values in Target
print("NaN in Target:", sp500_data["Target"].isna().sum())
print("\nTarget distribution (before train/test split):")
print(sp500_data["Target"].value_counts())
print(f"Percentage of 1s (price goes up): {sp500_data['Target'].mean()*100:.2f}%")


from sklearn.ensemble import RandomForestClassifier

# N estimators is the number of trees in the forest
# min_samples_split is the minimum number of samples required to split an internal node
# random_state is the random seed for reproducibility
model = RandomForestClassifier(n_estimators=100, min_samples_split=20, random_state=1)

train = sp500_data.iloc[:-100]
test = sp500_data.iloc[-100:]

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

# Predictors are the features we use to predict the target
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Diagnostic: Check feature statistics
print("\n" + "="*50)
print("FEATURE STATISTICS (Training Set):")
print(train[predictors].describe())

print("\n" + "="*50)
print("Training model...")
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])

print("\n" + "="*50)
print("PREDICTIONS:")
print(f"Predictions array: {preds}")
print(f"Unique predictions: {set(preds)}")
print(f"Number of 0s predicted: {(preds == 0).sum()}")
print(f"Number of 1s predicted: {(preds == 1).sum()}")

# Convert predictions to pandas Series with same index as test data
preds_series = pd.Series(preds, index=test.index, name="Predictions")

precision = precision_score(test["Target"], preds_series, zero_division=0)
print(f"\nPrecision Score: {precision:.4f}")

combined = pd.concat([test["Target"], preds_series], axis=1)
combined.plot()