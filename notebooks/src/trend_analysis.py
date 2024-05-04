import joblib
import pandas as pd
import matplotlib.pyplot as plt

def load_model(filepath):
    return joblib.load(filepath)

def forecast_trends(model, X):
    future_sales = model.predict(X)
    plt.figure(figsize=(10, 5))
    plt.plot(future_sales, label='Forecasted Sales')
    plt.title('Future Sales Trends')
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.savefig('../results/trend_forecasts.png')

if __name__ == "__main__":
    model = load_model('../models/trained_model.pkl')
    X_future
