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

X_train_scale = (X_train_values - X_mean)/ X_std
X_value_scaled = (X_val_values - X_mean)/ X_std

def sigmoid(z):
    return 1/ (1 +np.exp(-z))

m_train = X_train_scale.shape[0]
m_val = X_value_scaled.shape[0]

X_train_b = np.c_[np.ones((m_train, 1)), X_train_scale]
X_val_b = np.c_[np.ones((m_val, 1)), X_value_scaled]

def compute_log_loss(X, y, theta):
    z = X @ theta
    predictions = sigmoid(z)

    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    y = y.reshape(-1, 1)  # FORCE correct shape

    loss = -np.mean(
        y * np.log(predictions) +
        (1 - y) * np.log(1 - predictions)
    )
    return loss

def logistic_gradient_descent(X, y , learning_rate = 0.1, epochs = 1000):
    theta = np.random.randn(X.shape[1], 1) * 0.01
    loss_history = []
    y = y.reshape(-1, 1)
    
    for _ in range(epochs):
        z = X.dot(theta)
        predictions = sigmoid(z)
        gradients = (1/ len(y)) *X.T.dot(predictions - y)
        theta -= learning_rate * gradients
        loss = compute_log_loss(X, y, theta)
        loss_history.append(loss)
    return theta, loss_history


print("Unique y values:", np.unique(y_train.values))

theta_log, loss_history = logistic_gradient_descent(X_train_b, y_train.values, learning_rate=0.1, epochs=1000  )
print("Final Log Loss:", loss_history[-1])
print("Theta:", theta_log)

plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.title("Logistic Regression Training Loss")
plt.show()