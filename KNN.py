  
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter

#define the eucledian distance
def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))


class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]  
        # return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        #counter returns a tuple but we need to get the first arg
        return most_common[0][0]

#define the basic accuracy [0:1]
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

#get the demo data
iris = datasets.load_iris()
X, y = iris.data, iris.target

#get the resampling method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

k = 3
clf = KNN(k=k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("custom KNN classification accuracy", accuracy(y_test, predictions))
