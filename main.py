import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd

df = pd.DataFrame(
    {"Study Time": np.array([1, 2, 3, 4, 5, 6, 7, 8]),
     "Marks": np.array([20, 30, 40, 50, 60, 70, 80, 90])}
)

X = df[["Study Time"]]
y = df["Marks"]
model = LinearRegression()
model.fit(X,y)

new_hours = 5.5
predicted_marks = model.predict([[new_hours]])

print(f"{predicted_marks}")