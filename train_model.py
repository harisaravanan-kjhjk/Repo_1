# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Example dataset
data = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5],
    'x2': [2, 1, 2, 1, 2],
    'y': [5, 7, 9, 11, 13]
})

X = data[['x1', 'x2']]
y = data['y']

model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'regression_model.pkl')
print("Model saved!")
