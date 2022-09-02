import math


class Cost:
    def MSE(y, y_hat):
        assert len(y) == len(y_hat), "Number of elements does not match"

        n = len(y)
        err = 0

        for i in range(n):
            err += (y[i] - y_hat[i])**2

        return err / n
    
    def logistic(y, y_hat):
        assert len(y) == len(y_hat), "Number of elements does not match"
        
        EPSILON = 1e-10
        n = len(y)
        err = 0
        
        for i in range(n):
            # Avoid log(0)
            if y_hat[i] == 1 or y_hat[i] == 0: y_hat[i] = abs(y_hat[i] - EPSILON)
            
            err += -y[i] * math.log(y_hat[i]) - (1 - y[i]) * math.log(1 - y_hat[i])
            
        return err / n


class Activation:
    def sigmoid(z):
        return 1 / (1 + math.exp(-z))


class Optimizer:
    def gd(alpha, x, y, w, hyp):
        assert len(x) == len(y), "Number of elements does not match"
        assert len(w) == len(x[0]), "Number of parameters does not match with number of x values"

        n = len(y)
        m = len(w)

        for i in range(m):
            update = 0

            for j in range(n):
                err = hyp(w, x[j]) - y[j]
                
                update += err * x[j][i]

            update /= n  # Alternatively 2n

            w[i] = w[i] - alpha * update

        
class Hypothesis:
    def linear(w, x):
        assert len(x) == len(w), "Number of elements does not match"
        
        n = len(x)
        y = 0
        
        for i in range(n):
            y += w[i] * x[i]
            
        return y


import pandas as pd
import os
import random

data_dir = "data"

iris_header = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "class"
]
iris = pd.read_csv(os.path.join(data_dir, "iris.data"), names=iris_header)
iris.head()

iris_class = pd.unique(iris["class"])


def train_test_split(x, y, r=0.2):
    assert len(x) == len(y), "Number of elements must be equal in both dataframes"
    
    n = len(x)
    
    len_train = int(n * (1 - r))
    len_test = n - len_train
        
    idx_train = 0
    idx_test = 0
    
    x_train = [None] * len_train
    y_train = [None] * len_train
    
    x_test = [None] * len_test
    y_test = [None] * len_test
    
    ones = int(y.sum())  # The sum of all ones gives the number of right classifications
    zeros = n - ones  # The number of zeros is just the former subtraction
    
    ones_train = int(ones * (1 - r))
    ones_test = ones - ones_train
    
    zeros_train = int(zeros * (1 - r))
    zeros_test = zeros - zeros_train
    
    x = x.sample(frac=1)  # Random shuffling of samples
    
    for row in x.iterrows():
        idx = row[0]
        x_values = list(row[1])
        y_value = int(y.iloc[idx])
        
        if y_value == 1:
            if ones_train > 0:
                x_train[idx_train] = x_values
                y_train[idx_train] = y_value
                
                idx_train += 1
                ones_train -= 1
            elif ones_test > 0:
                x_test[idx_test] = x_values
                y_test[idx_test] = y_value
                
                idx_test += 1
                ones_test -= 1
        else:
            if zeros_train > 0:
                x_train[idx_train] = x_values
                y_train[idx_train] = y_value
                
                idx_train += 1
                zeros_train -= 1
            elif zeros_test > 0:
                x_test[idx_test] = x_values
                y_test[idx_test] = y_value
                
                idx_test += 1
                zeros_test -= 1
    
    return x_train, y_train, x_test, y_test


X_raw = iris.drop("class", axis=1)
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())  # Min-max scaling

X["_ones"] = [1] * len(X)  # Add ones-column for bias parameter

X_cols = len(X.columns)

models = [[0] * X_cols for _ in range(len(iris_class))]
errors = [[] for _ in range(len(iris_class))]

tests = [None] * len(models)
test_idx = 0

print("Training...")

for model, error, c in zip(models, errors, iris_class):
    Y = iris[["class"]] == c
    
    x_train, y_train, x_test, y_test = train_test_split(X, Y)

    tests[test_idx] = (x_test, y_test)  # Store the test sets
    
    err = 1
    epoch = 0
    EPOCHS = 1e5
    TOLERANCE = 1e-5
    alpha = 0.01
    
    while err > TOLERANCE and epoch < EPOCHS:
        hyp = Hypothesis.linear
        act = Activation.sigmoid
        
        Optimizer.gd(alpha, x_train, y_train, model, lambda w, x: act(hyp(w, x)))
        
        y_hat = [act(hyp(model, x)) for x in x_train]
        
        err = Cost.logistic(y_train, y_hat)
        error.append(err)
        
        epoch += 1

    test_idx += 1
        
    print(c, err, epoch, "Done!")

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for error, c in zip(errors, iris_class):
    ax.plot(error, label=c)
    
ax.legend()
plt.show()

