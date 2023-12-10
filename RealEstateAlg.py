import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Data cleaning
file_path = r"C:\Users\mahlo\OneDrive\Documents\RealEstateAlgorithm\Real estate.csv"
df = pd.read_csv(file_path)

# Handle missing values using median
df = df.fillna(df.median())

# Encode categorical variables, drop_first to avoid multicollinearity
df = pd.get_dummies(df, columns=['X1 transaction date'], drop_first=True)

# Feature engineering
# Check and handle infinite or NaN values in 'price_per_sqft'
df['price_per_sqft'] = df['Y house price of unit area'] / df['X2 house age']
df['price_per_sqft'].replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['price_per_sqft'], inplace=True)

# EDA
print(df.describe())
print(df.corr()['Y house price of unit area'])

# Visualization 
fig, ax = plt.subplots()
ax.scatter(df['X3 distance to the nearest MRT station'], df['Y house price of unit area'], c=df['X2 house age'])
ax.set_title("Price vs Distance to MRT")
plt.savefig('price_vs_distance.png')

# Display the scatter plot in a separate window
plt.show()


# ML model
X = df.drop('Y house price of unit area', axis=1)
y = df['Y house price of unit area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check for infinite or NaN values in features before fitting the model
if np.any(np.isfinite(X_train)) and np.any(np.isfinite(y_train)):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse}")
    print(f"R-squared: {r2}")

    # Save model
    with open('house_price_model.pkl', 'wb') as file:
        pickle.dump(model, file)
else:
    print("Input data contains infinite or NaN values. Please check and handle the issue.")
