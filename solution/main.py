import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data.csv")

# Split into features and target
X = data[['Area']]
y = data['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Take user input
area = int(input())

# Predict
predicted_price = model.predict([[area]])[0]

print(f"Predicted Price: {predicted_price:.2f} Lakhs")
