import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(filepath):
    return pd.read_csv(filepath)

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"MSE: {mean_squared_error(y_test, predictions)}")
    print(f"R^2: {r2_score(y_test, predictions)}")
    return model

if __name__ == "__main__":
    df = load_data('../data/featured_data.csv')
    X = df.drop('sold_count', axis=1)
    y = df['sold_count']
    model = train_model(X, y)
    joblib.dump(model, '../models/trained_model.pkl')
