import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('train.csv')

# Feature selection
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'
df = data[features + [target]].dropna()

# Define X and y
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# Coefficients
coeff_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
print(coeff_df)

# Predict function
def predict_price(sqft, bedrooms, bathrooms):
    return model.predict(pd.DataFrame([[sqft, bedrooms, bathrooms]], columns=features))[0]


print("Predicted price(₹):", predict_price(2000, 3, 2))
