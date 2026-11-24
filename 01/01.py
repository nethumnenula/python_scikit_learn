from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


X, y = fetch_california_housing(return_X_y=True)

model = KNeighborsRegressor().fit(X, y)
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=10))
])
pipe.fit(X, y)

pred = pipe.predict(X)

plt.scatter(pred, y)
plt.tight_layout()
plt.show()