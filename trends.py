import yfinance as yf
import pandas as pd
import numpy as np
from stock_features import StockFeatures
from sklearn.metrics import precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier


def get_ticker_input():
    ticker = input("Enter the ticker symbol: ").strip().upper()
    return ticker


def download_stock_data(ticker):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period="max")
    
    # del stock_data["Dividends"]
    # del stock_data["Stock Splits"]
    
    stock_data.plot.line(y="Close", use_index=True)
    
    return stock_data


def prepare_target_data(stock_data):
    stock_data["Tomorrow"] = stock_data["Close"].shift(-1)
    stock_data["Target"] = (stock_data["Tomorrow"] > stock_data["Close"]).astype(int)
    
    # Predicting the movement of the stock
    # Filter data from 1990 onwards
    stock_data = stock_data.loc["1975-01-01":].copy()
    
    return stock_data


def add_features(stock_data):
    # Add features using StockFeatures class
    features = StockFeatures()
    stock_data = features.features(stock_data)
    
    # Drop the last NaN row in Tomorrow 
    stock_data = stock_data.dropna()
    
    return stock_data, features


def print_diagnostics(stock_data):

    #--- AI Generated Code ---#
    # Diagnostic: Check for NaN values in Target
    print("NaN in Target:", stock_data["Target"].isna().sum())
    print("\nTarget distribution (before train/test split):")
    print(stock_data["Target"].value_counts())
    print(f"Percentage of 1s (price goes up): {stock_data['Target'].mean()*100:.2f}%")
    #--- AI Generated Code ---#


def create_model():
    # N estimators is the number of trees in the forest
    # min_samples_split is the minimum number of samples required to split an internal node
    # random_state is the random seed for reproducibility
    model = RandomForestClassifier(n_estimators=400, min_samples_split=50, max_depth=15, random_state=1, n_jobs=-1)
    return model


def split_train_test(stock_data):
    train = stock_data.iloc[:-100] 
    test = stock_data.iloc[-100:] # last 100 rows for testing
    return train, test


def print_train_diagnostics(train, test):
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


def get_predictors(features, stock_data):
    # Get predictor names 
    predictor_names = features.predictors()
    # get predictors that are in the data
    predictors = features.validate(stock_data, predictor_names)
    return predictors


def print_feature_statistics(train, predictors):
    #--- AI Generated Code ---#
    # Diagnostic: Check feature statistics
    print("\n" + "="*50)
    print("FEATURE STATISTICS (Training Set):")
    print(train[predictors].describe())
    
    print("\n" + "="*50)
    print("Training model...")
    #--- AI Generated Code ---#


def train_model(model, train, predictors):
    model.fit(train[predictors], train["Target"])


def make_predictions(model, test, stock_data, predictors):
    preds = model.predict(test[predictors])
    
    # Predict tomorrow's trend
    latest_features = stock_data[predictors].iloc[-1:].values # last row of the data
    tomorrow_prediction = model.predict(latest_features)[0] 
    tomorrow_probability = model.predict_proba(latest_features)[0]
    
    return preds, tomorrow_prediction, tomorrow_probability


def print_predictions(preds, tomorrow_prediction, tomorrow_probability):
    #--- AI Generated Code ---#
    print("\n" + "="*50)
    print("PREDICTIONS:")
    print(f"Predictions array: {preds}")
    print(f"Number of 0s predicted: {(preds == 0).sum()}")
    print(f"Number of 1s predicted: {(preds == 1).sum()}")
    print("\n" + "="*50)
    print("TOMORROW'S PREDICTION:")
    print(f"Will stock go UP or DOWN tomorrow: {'UP' if tomorrow_prediction == 1 else 'DOWN'}")
    print(f"Confidence: {max(tomorrow_probability)*100:.2f}%")
    print(f"Probability of UP: {tomorrow_probability[1]*100:.2f}%")
    print(f"Probability of DOWN: {tomorrow_probability[0]*100:.2f}%")
    #--- AI Generated Code ---#


def calculate_metrics(test, preds):
    # Convert predictions to pandas Series with same index as test data
    preds_series = pd.Series(preds, index=test.index, name="Predictions")
    
    # Calculate precision and accuracy
    precision = precision_score(test["Target"], preds_series, zero_division=0)
    accuracy = accuracy_score(test["Target"], preds_series)
    
    return precision, accuracy


def print_metrics(accuracy, precision):
    #--- AI Generated Code ---#
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS:")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%) - Overall correctness of ALL predictions")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%) - Does not calculate true negatives")
    print("\n" + "="*50)
    #--- AI Generated Code ---#


def main():

    ticker = get_ticker_input()
    
    stock_data = download_stock_data(ticker)
    
    stock_data = prepare_target_data(stock_data)
    
    stock_data, features = add_features(stock_data)
    
    print_diagnostics(stock_data)
    
    model = create_model()
    
    train, test = split_train_test(stock_data)
    
    print_train_diagnostics(train, test)
    
    predictors = get_predictors(features, stock_data)
    
    print_feature_statistics(train, predictors)
    
    train_model(model, train, predictors)
    
    preds, tomorrow_prediction, tomorrow_probability = make_predictions(model, test, stock_data, predictors)
    
    print_predictions(preds, tomorrow_prediction, tomorrow_probability)
    
    precision, accuracy = calculate_metrics(test, preds)
    
    print_metrics(accuracy, precision)


if __name__ == "__main__":
    main()
