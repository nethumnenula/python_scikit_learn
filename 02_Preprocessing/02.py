# weaknesses of StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

x = np.random.exponential(10, (1000)) + np.random.normal(0, 1, (1000))

# plt.hist(x, 30)
plt.hist((x - np.mean(x))/np.std(x), 30)
plt.show()