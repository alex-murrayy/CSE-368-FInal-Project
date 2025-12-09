import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(y_true, y_pred):
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # Directional accuracy (1 if model predicts up/down correctly)
    true_direction = np.sign(np.diff(y_true.values))
    pred_direction = np.sign(np.diff(y_pred))
    direction_acc = (true_direction == pred_direction).mean()
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Directional Accuracy: {direction_acc*100:.2f}%")
    
    return rmse, mae, direction_acc


def next_day_price(ticker):
    df = yf.download(ticker, period="2y")

    df["Return1d"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Volatility"] = df["Close"].pct_change().rolling(10).std()


    # target
    df["NextClose"] = df["Close"].shift(-1)
    df = df.dropna()

    train_size = int(0.8 * len(df))
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    features = ["Close", "Return1d", "MA5", "MA20", "Volatility"]
    X_train, y_train = train[features], train["NextClose"]
    X_test, y_test = test[features], test["NextClose"]

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    # KNN Regression with scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    knn_preds = knn.predict(X_test_scaled)  # Use scaled test features

    # Scale latest input for KNN
    latest_input = df[features].iloc[-1].values.reshape(1, -1)
    latest_input_scaled = scaler.transform(latest_input)

    lr_next_day_price = lr.predict(latest_input)[0]
    knn_next_day_price = knn.predict(latest_input_scaled)[0]
    knn_next_day_price = np.clip(knn_next_day_price, y_train.min(), y_train.max())


    print("Predicted next close (LR):", lr_next_day_price)
    print("Predicted next close (KNN):", knn_next_day_price)
    print(evaluate_model(y_test, lr_preds))

    return lr_next_day_price

# next_day_price("AAPL")


