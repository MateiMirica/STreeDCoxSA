from pystreed import STreeDCoxSurvivalAnalysis
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

data = pd.read_csv("data/cox-survival-analysis/UnempDur.csv", delimiter=",").fillna(0)
data = np.array(data)
x_categ_columns = [8]
x_categ_columns2 = [5]
x_cont_columns = [3, 4, 5, 6, 7, 8]

for c in x_categ_columns:
    dict = {}
    val = 0
    for i in range(len(data)):
        if not (data[i][c] in dict):
            dict[data[i][c]] = val
            val = val + 1
        data[i][c] = dict[data[i][c]]

# Create the y array with 'time' and 'event'
y = np.array(data[:, [2, 1]])

# Select columns for the x array
X = np.array(data[:, x_cont_columns])

times = np.array(y[:, 0])
events = []

for i in range(len(y)):
    if y[i][1] == 1:
        events.append(True)
    else:
        events.append(False)

events = np.array(events)

# Train an optimal cox survival tree model
model = STreeDCoxSurvivalAnalysis(max_depth=2, max_num_nodes=2, l1_ratio=0.4, hyper_tune=True)

model.fit(X, y)

# Measure the performance of the model
prediction = model.predict(X)
prediction = -prediction
result = concordance_index_censored(events, times, prediction)
print("Harrell's concordance index: ", result[0])
print("Objective score: ", model.score(X, y))
