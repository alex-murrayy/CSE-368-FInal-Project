import yfinance as yf
import pandas as pd
import numpy as np
from stock_features import StockFeatures
from sklearn.metrics import precision_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

class StockTrendPredictor:
    def __init__(self, ticker):

        self.ticker = ticker.strip().upper()
        self.stock_data = None
        self.model = None
        self.features = None
        self.predictors = None
        self.train = None
        self.test = None
        self.preds = None
        self.tomorrow_prediction = None
        self.tomorrow_probability = None
        self.accuracy = None
        self.precision = None

    def download_stock_data(self):

        stock = yf.Ticker(self.ticker)
        self.stock_data = stock.history(period="max")
        
        # del stock_data["Dividends"]
        # del stock_data["Stock Splits"]
        
        self.stock_data.plot.line(y="Close", use_index=True)

    def prepare_target_data(self):

        self.stock_data["Tomorrow"] = self.stock_data["Close"].shift(-1)
        self.stock_data["Target"] = (self.stock_data["Tomorrow"] > self.stock_data["Close"]).astype(int)
        
        # Predicting the movement of the stock
        # Filter data from 1990 onwards
        self.stock_data = self.stock_data.loc["1975-01-01":].copy()

    def add_features(self):

        # Add features using StockFeatures class
        self.features = StockFeatures()
        self.stock_data = self.features.features(self.stock_data)
        
        # Drop the last NaN row in Tomorrow 
        self.stock_data = self.stock_data.dropna()

    def print_diagnostics(self):

        #--- AI Generated Code ---#
        # Diagnostic: Check for NaN values in Target
        print("NaN in Target:", self.stock_data["Target"].isna().sum())
        print("\nTarget distribution (before train/test split):")
        print(self.stock_data["Target"].value_counts())
        print(f"Percentage of 1s (price goes up): {self.stock_data['Target'].mean()*100:.2f}%")
        #--- AI Generated Code ---#

    def create_model(self):

        # N estimators is the number of trees in the forest
        # min_samples_split is the minimum number of samples required to split an internal node
        # random_state is the random seed for reproducibility
        self.model = RandomForestClassifier(n_estimators=400, min_samples_split=50, max_depth=15, random_state=1, n_jobs=-1)

    def split_train_test(self):

        self.train = self.stock_data.iloc[:-100] 
        self.test = self.stock_data.iloc[-100:] # last 100 rows for testing

    def print_train_diagnostics(self):

        #--- AI Generated Code ---#
        # Diagnostic: Check class distribution in train and test sets
        print("\n" + "="*50)
        print("TRAINING SET:")
        print("Target distribution:", self.train["Target"].value_counts())
        print(f"Percentage of 1s: {self.train['Target'].mean()*100:.2f}%")
        print(f"Training samples: {len(self.train)}")
        
        print("\nTEST SET:")
        print("Target distribution:", self.test["Target"].value_counts())
        print(f"Percentage of 1s: {self.test['Target'].mean()*100:.2f}%")
        print(f"Test samples: {len(self.test)}")
        #--- AI Generated Code ---#

    def get_predictors(self):

        # Get predictor names 
        predictor_names = self.features.predictors()
        # get predictors that are in the data
        self.predictors = self.features.validate(self.stock_data, predictor_names)

    def print_feature_statistics(self):

        #--- AI Generated Code ---#
        # Diagnostic: Check feature statistics
        print("\n" + "="*50)
        print("FEATURE STATISTICS (Training Set):")
        print(self.train[self.predictors].describe())
        
        print("\n" + "="*50)
        print("Training model...")
        #--- AI Generated Code ---#

    def train_model(self):

        self.model.fit(self.train[self.predictors], self.train["Target"])

    def make_predictions(self):

        self.preds = self.model.predict(self.test[self.predictors])
        
        # Predict tomorrow's trend
        latest_features = self.stock_data[self.predictors].iloc[-1:].values # last row of the data
        self.tomorrow_prediction = self.model.predict(latest_features)[0] 
        self.tomorrow_probability = self.model.predict_proba(latest_features)[0]

    def print_predictions(self):

        #--- AI Generated Code ---#
        print("\n" + "="*50)
        print("PREDICTIONS:")
        print(f"Predictions array: {self.preds}")
        print(f"Number of 0s predicted: {(self.preds == 0).sum()}")
        print(f"Number of 1s predicted: {(self.preds == 1).sum()}")
        print("\n" + "="*50)
        print("TOMORROW'S PREDICTION:")
        print(f"Will stock go UP or DOWN tomorrow: {'UP' if self.tomorrow_prediction == 1 else 'DOWN'}")
        print(f"Confidence: {max(self.tomorrow_probability)*100:.2f}%")
        print(f"Probability of UP: {self.tomorrow_probability[1]*100:.2f}%")
        print(f"Probability of DOWN: {self.tomorrow_probability[0]*100:.2f}%")
        #--- AI Generated Code ---#

    def calculate_metrics(self):

        # Convert predictions to pandas Series with same index as test data
        preds_series = pd.Series(self.preds, index=self.test.index, name="Predictions")
        
        # Calculate precision and accuracy
        self.precision = precision_score(self.test["Target"], preds_series, zero_division=0)
        self.accuracy = accuracy_score(self.test["Target"], preds_series)

    def print_metrics(self):

        #--- AI Generated Code ---#
        print("\n" + "="*50)
        print("MODEL PERFORMANCE METRICS:")
        print(f"Accuracy:  {self.accuracy:.4f} ({self.accuracy*100:.2f}%) - Overall correctness of ALL predictions")
        print(f"Precision: {self.precision:.4f} ({self.precision*100:.2f}%) - Does not calculate true negatives")
        print("\n" + "="*50)
        #--- AI Generated Code ---#

    def predict(self):
        
        self.download_stock_data()
        self.prepare_target_data()
        self.add_features()
        self.print_diagnostics()
        self.create_model()
        self.split_train_test()
        self.print_train_diagnostics()
        self.get_predictors()
        self.print_feature_statistics()
        self.train_model()
        self.make_predictions()
        self.print_predictions()
        self.calculate_metrics()
        self.print_metrics()
