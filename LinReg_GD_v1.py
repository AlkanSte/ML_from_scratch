import numpy as np

class LinearRegression:

    def __init__(self, lr = 0.001, n_iters = 1000):

        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #init our parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated


def RMSE(y_true, y_pred):
    return np.mean(np.sqrt((y_true - y_pred) ** 2))

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
    


from sklearn import datasets

# from linear_regression import LinearRegression
# from regression import LinearRegression


X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=142)
#Now lets test our algo!

#Data shuffling primarily for train_test_split function
def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

#train_test_split function that mimics sklearn function
def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


#Run the train test plit
train_test_split(X, y, test_size=0.7, shuffle=True, seed=142):

regressor = LinearRegression(lr=0.05, n_iters=100)
regressor.fit(X_train, y_train)
predictions =regressor.predict(X_test)

mse = MSE(y_test, predictions)
rmse = RMSE(y_test, predictions)
print("Our custom LinReg MSE scored value:", mse)
print("Our custom LinReg RMSE scored value:", rmse)
