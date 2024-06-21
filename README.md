[![CMake build](https://github.com/algtudelft/pystreed/actions/workflows/cmake.yml/badge.svg)](https://github.com/algtudelft/pystreed/actions/workflows/cmake.yml)
[![Pip install](https://github.com/algtudelft/pystreed/actions/workflows/pip.yml/badge.svg)](https://github.com/algtudelft/pystreed/actions/workflows/pip.yml)

# STreeD: Separable Trees with Dynamic programming

## Python usage

### Install from PyPi
The `pystreed` python package can be installed from PyPi using `pip`:

```sh
pip install pystreed
```

### Install from source using pip
The `pystreed` python package can also be installed from source as follows:

```sh
git clone https://github.com/AlgTUDelft/pystreed.git
cd pystreed
pip install . 
```

### Example usage
`pystreed` can be used, for example, as follows (make sure the file is present):

```python
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

```


See the [examples](examples) folder for a number of example usages.

Note that some of the examples require the installation of extra python packages:

```sh
pip install matplotlib seaborn graphviz scikit-survival pydl8.5 pymurtree
```

Note that `pymurtree` is currently not available for pip install yet. It can be installed from [source](https://github.com/MurTree/pymurtree/) (install the `develop` branch)

Graphviz additionaly requires another instalation of a binary. See [their website](https://graphviz.org/download/).

## C++ usage

### Compiling
The code can be compiled on Windows or Linux by using cmake. For Windows users, cmake support can be installed as an extension of Visual Studio and then this repository can be imported as a CMake project.

For Linux users, they can use the following commands:

```sh
mkdir build
cd build
cmake ..
cmake --build .
```
The compiler must support the C++17 standard

### Running
After STreeD is built, the following command can be used (for example):
```sh
./STreeD -task accuracy -file ../data/cost-sensitive/car/car-train-1.csv -max-depth 3 -max-num-nodes 7
```

Run the program without any parameters to see a full list of the available parameters.

### Docker
Alternatively, docker can be used to build and run STreeD:
```
docker build -t streed .
docker container run -it streed /STreeD/build/STREED -task accuracy -file /STreeD/data/cost-sensitive/car/car-train-1.csv -max-depth 3 -max-num-nodes 7
```
