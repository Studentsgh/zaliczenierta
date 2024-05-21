import numpy as np
from sklearn.linear_model import Perceptron

np.random.seed(0)

X = np.array([
    [1, 1],
    [3, 3],
    [5, 4],
    [7, 5],
])

y = np.array([
    0,  
    1,  
    0, 
    1,  
])


np.random.shuffle(X)
np.random.shuffle(y)


X_train = X[:2]
y_train = y[:2]
X_test = X[2:]
y_test = y[2:]

model = Perceptron()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predykcje dla danych testowych:")

print(y_pred)


