import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

array = np.array(["low", "low", "high", "medium"]).reshape(-1, 1)

print(array)

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ss = encoder.fit_transform(array)
print(ss)

zero = encoder.transform([["zero"]])
print(zero)

