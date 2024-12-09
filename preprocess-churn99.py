import os
import numpy as np
import pandas as pd

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    
    # Read Data
    df = pd.read_csv(f"{base_dir}/input/human_activity11.csv")
    
    # Drop Rows with Null Values
    df = df.dropna()
    
    # Map the label column to activity type (1 = walking, 6 = standing)
    df['activity_type'] = df['label'].map({1: 'walking', 6: 'standing'})
    
    # Drop the original 'label' column
    df.drop(['label'], axis=1, inplace=True)
    
    # Apply one hot encoding on 'activity_type'
    df = pd.get_dummies(df, prefix=['activity'], columns=['activity_type'])
    
    # Split into features (X) and target (y)
    y = df.pop("activity_walking")  # We will predict the walking activity
    X = df
    
    # Convert y to numpy array and reshape
    y_pre = y.to_numpy().reshape(len(y), 1)
    X = np.concatenate((y_pre, X), axis=1)
    
    # Shuffle the data
    np.random.shuffle(X)
    
    # Split into Train, Validation, and Test datasets
    train, validation, test = np.split(X, [int(.7 * len(X)), int(.85 * len(X))])
    
    # Convert to DataFrames
    train = pd.DataFrame(train)
    validation = pd.DataFrame(validation)
    test = pd.DataFrame(test)
    
    # Convert the label column to integer (for model fitting)
    train[0] = train[0].astype(int)
    validation[0] = validation[0].astype(int)
    test[0] = test[0].astype(int)
    
    # Save the DataFrames as CSV files
    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    
    # For Machine Learning (Model fitting and evaluation)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Split X and y for training
    X_train = train.drop(columns=0)
    y_train = train[0]
    X_test = test.drop(columns=0)
    y_test = test[0]
    
    # Initialize and train Random Forest model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
