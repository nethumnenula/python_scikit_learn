import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, QuantileTransformer

def plot_output(scaler):
    X_scaled = scaler.fit_transform(X)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y)
    plt.title(type(scaler).__name__)

df = pd.read_csv("data.csv")
print(df)

X = df[['x', 'y']].values
y = df['z'] == 'c'

plt.figure()
# Before Scaling
# plt.scatter(X[:, 0], X[:, 1], c=y)

plt.figure()
# After Scaling
# X_new = StandardScaler().fit_transform(X)
X_new = QuantileTransformer(n_quantiles=400).fit_transform(X)
# plt.scatter(X_new[:, 0], X_new[:, 1], c=y)

plot_output(scaler=StandardScaler()) # This is not a built-in method

plt.show()