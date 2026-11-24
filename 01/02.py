from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

model = LinearRegression()

model.fit(X, y)

print(model.predict(X))