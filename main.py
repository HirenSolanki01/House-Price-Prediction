import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("train.csv")

# Select relevant features and target
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].dropna()

# Define input features and output
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("\n--- Model Evaluation ---")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# Display feature names and coefficients
print("\n--- Model Coefficients ---")
feature_names = ['Living Area (sqft)', 'Bedrooms', 'Bathrooms']
for name, coef in zip(feature_names, model.coef_):
    print(f"{name:<20}: {coef:.2f}")

# Bonus: Get validated input from user
print("\n--- Give details about house ---")

def get_positive_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value < 0:
                print("❌ Please enter a positive number.")
            else:
                return value
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

def get_positive_int(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value < 0:
                print("❌ Please enter a non-negative integer.")
            else:
                return value
        except ValueError:
            print("❌ Invalid input. Please enter an integer.")

# Take input from the user
input_area = get_positive_float("Enter Living Area (in sqft): ")
input_bedrooms = get_positive_int("Enter number of Bedrooms: ")
input_bathrooms = get_positive_int("Enter number of Full Bathrooms: ")

# Predict using the trained model
user_input = pd.DataFrame([[input_area, input_bedrooms, input_bathrooms]],
                          columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
predicted_price = model.predict(user_input)[0]

print(f"\nEstimated House Price: ₹{predicted_price:,.2f}")
