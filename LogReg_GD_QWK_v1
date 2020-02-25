import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    #our X is an numpy nd vector of size m*n.
    #Where n is the number of samples and n is the number of features
    #y is 1d vector of size n
    def fit(self, X, y):
        #first init the parameters
        #First D is the number of samples, second - features
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        #we can also use random number generator for first weights first weights initialization
        self.bias = 0

        #set the gradient descent to iteratively update our weights
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            #now we need to update our weights
            #we first calc the derivatives

            dw = (1 / n_samples * np.dot(X.T, (y_predicted - y)))
            db = (1 / n_samples * np.sum(y_predicted - y))

            #update
            self.weights -= self.learning_rate *dw
            self.bias -= self.learning_rate * db
            
    def _sigmoid(self, x):
        #pass the sigmoid function s(x) = 1/1+e**-wx+b
        return 1/ (1+ np.exp(-x))
 

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_class


#define Quadratic Weighted Kappa (https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps)
#we will use the Ultrafast method props to (https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-657027)

from numba import jit
@jit


def qwk3(a1, a2, max_rat=3):
    assert(len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e

#define accuracy score
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


#Now lets test our algo!
from sklearn.model_selection import train_test_split
from sklearn import datasets

bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 142)

#Run the regression
regressor = LogisticRegression(learning_rate = 0.001, n_iters = 1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print('LR accuracy:', accuracy(y_test, predictions))
print('LR QWK:', qwk3(y_test, predictions))

