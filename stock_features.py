import pandas as pd
import numpy as np

# Technical Analysis Library : https://github.com/bukosabino/ta
from ta.trend import SMAIndicator
from ta.momentum import ROCIndicator


class StockFeatures:
    
    def __init__(self):
        pass
    
    def features(self, stock_data):
        
        data = stock_data.copy()
        
        # High-Low Ratio : https://www.investopedia.com/terms/t/truerange.asp
        data["High_Low_Ratio"] = data["High"] / data["Low"]
        
        # Close-Open Ratio : https://www.investopedia.com/trading/candlestick-charting-what-is-it/
        data["Close_Open_Ratio"] = data["Close"] / data["Open"]
        
        # Simple Moving Average : https://www.investopedia.com/terms/s/sma.asp, TA LIB: https://ta-lib.org/function.html
        data["MA_5"] = SMAIndicator(close=data["Close"], window=5).sma_indicator()
        data["MA_10"] = SMAIndicator(close=data["Close"], window=10).sma_indicator()
        data["MA_20"] = SMAIndicator(close=data["Close"], window=20).sma_indicator()
        
        # Price vs Moving Average : https://www.investopedia.com/terms/p/price-action.asp
        data["Price_vs_MA5"] = (data["Close"] / data["MA_5"]) - 1
        data["Price_vs_MA10"] = (data["Close"] / data["MA_10"]) - 1
        data["Price_vs_MA20"] = (data["Close"] / data["MA_20"]) - 1
        
        # Moving Average Crossover : https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp
        data["MA5_vs_MA20"] = (data["MA_5"] / data["MA_20"]) - 1
        
        # Volume Ratio : https://www.investopedia.com/terms/v/volume.asp, https://www.investopedia.com/terms/v/volume-rate-change-vroc.asp
        data["Volume_MA20"] = data["Volume"].rolling(window=20).mean()
        data["Volume_Ratio"] = data["Volume"] / data["Volume_MA20"]
        
        # Rate of Change (ROC) : https://www.investopedia.com/terms/r/rateofchange.asp, TA LIB: https://ta-lib.org/function.html
        data["Price_Change_3d"] = ROCIndicator(close=data["Close"], window=3).roc() / 100
        
        return data
    
    def predictors(self):
        # Return list of feature names to use for prediction
        return [
            "High_Low_Ratio",
            "Close_Open_Ratio",
            "Price_vs_MA5",
            "Price_vs_MA10",
            "Price_vs_MA20",
            "MA5_vs_MA20",
            "Volume_Ratio",
            "Price_Change_3d"
        ]
    
    def validate(self, stock_data, predictor_names):
        # Return only predictors that exist in the data 
        available = [p for p in predictor_names if p in stock_data.columns]
        
        return available

