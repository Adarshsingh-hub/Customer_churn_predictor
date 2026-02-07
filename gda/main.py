import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\ADARSH SINGH\\customer_Churn_predictor\\data\\telco_Churn.csv")

#print(data.head())
#print(data.info())

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')
data = data.dropna()

features = ["tenure", "MonthlyCharges", "TotalCharges"]
X = data[features]
y = data["Churn"].map({"Yes": 1, "No": 0})
print(y.value_counts())


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state= 42, stratify = y)

X_train_values = X_train.values
X_val_values = X_val.values

X_mean = np.mean(X_train_values, axis=0)
X_std = np.std(X_train_values, axis = 0)

X_train_scaled = (X_train_values - X_mean)/ X_std
X_val_scaled = (X_val_values - X_mean)/ X_std

y_train_np = y_train.values
y_val_np = y_val.values

phi = np.mean(y_train_np)


mu_0 = np.mean(X_train_scaled[y_train_np == 0], axis=0)
mu_1 = np.mean(X_train_scaled[y_train_np == 1], axis=0)


def compute_shared_covariance(X, y, mu_0, mu_1):
    n_features = X.shape[1]
    sigma = np.zeros((n_features, n_features))
    
    for i in range(len(X)):
        if y[i] == 0:
            diff = (X[i] - mu_0).reshape(-1, 1)
        else:
            diff = (X[i] - mu_1).reshape(-1, 1)

        sigma += diff @ diff.T

    return sigma / len(X)

sigma = compute_shared_covariance(X_train_scaled, y_train_np, mu_0, mu_1)


def gaussian_log_pdf(x, mu, sigma):
    n = len(mu)
    diff = x - mu
    sigma_inv = np.linalg.inv(sigma)

    return -0.5 * (
        diff @ sigma_inv @ diff +
        np.log(np.linalg.det(sigma)) +
        n * np.log(2 * np.pi)
    )

def gda_predict(X, mu_0, mu_1, sigma, phi):
    preds = []

    for x in X:
        log_p0 = gaussian_log_pdf(x, mu_0, sigma) + np.log(1 - phi)
        log_p1 = gaussian_log_pdf(x, mu_1, sigma) + np.log(phi)

        preds.append(1 if log_p1 > log_p0 else 0)

    return np.array(preds)

y_val_pred_gda = gda_predict(
    X_val_scaled, mu_0, mu_1, sigma, phi
)

accuracy = np.mean(y_val_pred_gda == y_val_np)
print("GDA Validation Accuracy:", accuracy)

def precision_recall(y_true, y_pred):
    y_true = y_true.values.reshape(-1, 1)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-15)
    recall = tp / (tp + fn + 1e-15)

    return precision, recall


p, r = precision_recall(y_val, y_val_pred_gda.reshape(-1, 1))
print("GDA Precision:", p)
print("GDA Recall:", r)
