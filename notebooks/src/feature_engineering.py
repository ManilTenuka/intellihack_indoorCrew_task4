import pandas as pd

def extract_features(df):
    # Example: Create new features like price difference
    df['price_diff'] = df['original_price'] - df['current_price']
    return df

if __name__ == "__main__":
    df = pd.read_csv('../data/cleaned_data.csv')  # Assume this is loaded post-initial preprocessing
    df = extract_features(df)
    df.to_csv('../data/featured_data.csv', index=False)
