import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

X, y  = fetch_california_housing(return_X_y=True)
print(fetch_california_housing()['DESCR'])