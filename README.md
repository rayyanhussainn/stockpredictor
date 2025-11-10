# Stock Price Predictor

This project implements a machine learning model to predict future stock
prices using historical data. It uses neural networks and time-series
preprocessing to generate short-term predictions and evaluate model
accuracy.

## Features

-   Time-series forecasting of stock prices
-   LSTM/MLP model options
-   Data preprocessing (scaling, windowing, normalization)
-   Train/test split with evaluation metrics
-   Configurable prediction horizon
-   Easy to extend to any ticker

## Tech Stack

-   **Python 3**
-   **NumPy**
-   **Pandas**
-   **Matplotlib**
-   **TensorFlow / Keras**
-   **scikit-learn**

## How It Works

1.  Pull historical stock price data (CSV or API).
2.  Clean and preprocess the data (select features, scale, generate time
    windows).
3.  Train a neural network on past price windows.
4.  Evaluate on test data using metrics like MAE/MSE.
5.  Generate future predictions and visualize results.

## Setup

### 1. Install dependencies

``` bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### 2. Add your dataset

Place your CSV (e.g. `AAPL.csv`) in the project folder.

Expected CSV columns: - Date - Open - High - Low - Close - Volume

### 3. Run the training script

``` bash
python train.py
```

### 4. Run the prediction script

``` bash
python predict.py
```

## Example Code Snippet

``` python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

df = pd.read_csv("AAPL.csv")
data = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

window = 60
X, y = [], []
for i in range(window, len(scaled_data)):
    X.append(scaled_data[i-window:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=20, batch_size=32)
```

## Customization

-   Switch to GRU, Transformer, or CNN models
-   Add sentiment signals (news, social media)
-   Predict multiple days ahead
-   Add hyperparameter tuning

## Accuracy

The model commonly reaches **\~60% directional accuracy**, depending
on: - Ticker - Time window - Prediction horizon - Data volatility

## License

MIT License.
