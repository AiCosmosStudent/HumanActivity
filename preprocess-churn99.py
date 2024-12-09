import os
import numpy as np
import pandas as pd
import tempfile

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    
    # Read Data
    df = pd.read_csv(f"{base_dir}/input/human activity11.csv")

    # Drop rows with Null Values (handle missing data)
    df = df.dropna()

    # Feature Engineering: Generate statistical features for each accelerometer axis
    def generate_statistical_features(df):
        features = pd.DataFrame()
        
        # Loop through each accelerometer column and compute the statistics
        for col in ['back-x', 'back-y', 'back-z', 'thigh-x', 'thigh-y', 'thigh-z']:
            features[f'{col}_mean'] = [df[col].mean()]
            features[f'{col}_std'] = [df[col].std()]
            features[f'{col}_max'] = [df[col].max()]
            features[f'{col}_min'] = [df[col].min()]
            features[f'{col}_range'] = [df[col].max() - df[col].min()]
        
        return features

    # Generate statistical features for the accelerometer data
    features = generate_statistical_features(df)

    # Extract label column and encode as target
    y = df['label'].values  # Target variable (1 for walking, 6 for standing)

    # Combine the features and target variable
    X = np.concatenate([features.values, y.reshape(-1, 1)], axis=1)

    # Shuffle the data
    np.random.shuffle(X)

    # Split into Train, Validation, and Test datasets (70%, 15%, 15%)
    train, validation, test = np.split(X, [int(.7 * len(X)), int(.85 * len(X))])

    # Convert to DataFrame for better handling
    train = pd.DataFrame(train)
    validation = pd.DataFrame(validation)
    test = pd.DataFrame(test)

    # Convert the label column to integer
    train[0] = train[0].astype(int)
    validation[0] = validation[0].astype(int)
    test[0] = test[0].astype(int)

    # Save the processed data as CSV files
    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

    print("Feature Engineering Complete. Processed Data Saved.")
