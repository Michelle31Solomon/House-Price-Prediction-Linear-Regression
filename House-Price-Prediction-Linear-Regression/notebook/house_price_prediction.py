import pandas as pd

data = pd.read_csv(
    r"C:\Users\Admin\OneDrive\Desktop\House-Price-Prediction-Linear-Regression\data\housing.csv"
)

print(data.head())
print(data.columns)

# STEP 4.2: Separate features and target
X = data.drop('price', axis=1)
y = data['price']

print("X shape:", X.shape)
print("y shape:", y.shape)

# STEP 4.3: Convert yes/no columns to numeric
binary_cols = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

for col in binary_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})

# Convert furnishing status to numbers
data['furnishingstatus'] = data['furnishingstatus'].map({
    'unfurnished': 0,
    'semi-furnished': 1,
    'furnished': 2
})

print(data.head())
X = data.drop('price', axis=1)
y = data['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
from sklearn.linear_model import LinearRegression

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model training completed")
# Predict on test data
y_pred = model.predict(X_test)

print(y_pred[:5])

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices (Linear Regression)")
plt.show()


