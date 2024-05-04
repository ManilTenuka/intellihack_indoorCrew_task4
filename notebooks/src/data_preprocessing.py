import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    return df

def encode_categorical(df):
    # Convert categorical variables using pd.get_dummies
    return pd.get_dummies(df, drop_first=True)

def normalize_data(df):
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler

if __name__ == "__main__":
    df = load_data('../data/Watches_Bags_Accessories.csv')
    df = clean_data(df)
    df = encode_categorical(df)
    df, scaler = normalize_data(df)
