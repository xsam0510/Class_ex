import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut


def Naive_Bayes_model(dev):
    # Encode the target variable
    le = LabelEncoder()
    dev['group'] = le.fit_transform(dev['group'])

    X, y = dev.drop('group', axis=1), dev['group']

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Create a Gaussian Naive Bayes model
    nb = GaussianNB()

    # Perform Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    
    # Keep track of validation performance for each fold
    accuracies = []
    for train_idx, val_idx in loo.split(X):
        # Split the data into training and validation sets
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit the model to the training data
        nb.fit(X_train, y_train)

        # Make predictions on the validation data
        y_pred = nb.predict(X_val)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)

    # Calculate the mean accuracy
    mean_accuracy = np.mean(accuracies)
    return print("Mean Accuracy:", mean_accuracy)