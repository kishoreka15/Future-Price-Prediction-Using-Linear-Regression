import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# Dataset
x = np.array([2005, 2010, 2015, 2020, 2025]).reshape(-1, 1)

y = np.array([5.5, 8.25, 8.25, 13.75, 16.99])
# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
# Train Model
model = LinearRegression()

model.fit(x_train, y_train)
# Training Data
print("\n===== TRAINING DATA =====")
print(x_train)

print("\n===== LABEL DATA =====")
print(y_train)
# Test Prediction
y_pred = model.predict(x_test)

print("\n===== TEST PREDICTION =====")
print(y_pred)
# Error Metric
mae = mean_absolute_error(y_test, y_pred)

print(f"\nMean Absolute Error : {mae:.2f}")
# Future Predictions

future_years = np.array([2030, 2035, 2040]).reshape(-1, 1)

future_predictions = model.predict(future_years)

print("\n===== FUTURE PREDICTIONS =====")

for year, price in zip(future_years, future_predictions):
    print(f"Year {year[0]} --> Predicted Price = {price:.2f}")
# User Input Prediction
user_year = int(input("\nEnter Future Year : "))

user_prediction = model.predict([[user_year]])

print(f"Predicted Price in {user_year} = {user_prediction[0]:.2f}")
# Graph Visualization

# Actual Data
plt.scatter(
    x,
    y,
    color="blue",
    s=100,
    label="Actual Data"
)

# Regression Line
plt.plot(
    x,
    model.predict(x),
    color="red",
    linewidth=2,
    label="Regression Line"
)

# Future Predictions
plt.scatter(
    future_years,
    future_predictions,
    color="green",
    s=100,
    label="Future Predictions"
)

# Labels and Title
plt.title("Price Prediction using Linear Regression")

plt.xlabel("Year")

plt.ylabel("Price")

plt.grid(True)

plt.legend()

# Show Graph
plt.show()
