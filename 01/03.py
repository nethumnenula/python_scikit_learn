import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


X, y = fetch_california_housing(return_X_y=True)


pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=1))
])
# pipe.get_params()
model = GridSearchCV(estimator=pipe,
                   param_grid={"model__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                   cv=3) # Cross Validation

model.fit(X, y)
# pred = model.predict(X)
df = pd.DataFrame(model.cv_results_)
# print(df)
# plt.scatter(pred, y)
# plt.tight_layout()
# plt.show()

