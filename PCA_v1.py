#PCA algo to get lineary independent features and recude the dimensionality of the dataset
#also with it we can rank reduced dimensionality according to the variance of data along  them

#we first define the variance by: Var(X) = 1/n * sum(Xi-mean.Xi)**2

#Here we willl also need the covariance matrix. It indicates the level to which two variables vary together
#This matrix can also be standartized, guess what we get then?)
#Cov(X,Y) = 1/n * sum(Xi - mean.Xi)*(Yi - mean.Yi)T
#Cov(X,X) = 1/n * sum(Xi - mean.Xi)*(Yi - mean.Xi)T

#We will also calc the rotation with the help of the eigenvectors. The corresponding eigenvalues indicate the importance of its corresponding eigen vector

#Our approach: 1. Subtract the mean from X and calculate the Cov(X,X)
#Calc the eigenvectors and eigenvalues of a Cov(X,X)
#Sort the eigenvectors according to eigenvalues,  choose k eigenvectors -> this is our k dimensions
#Transform the original data fron n to k dimensions

import numpy as np

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        #mean
        self.mean = np.mean(X, axis = 0)
        X = X - self.mean

        #covariance
        #row = 1 sample. Column = 1 feature
        #np.cov does it the other way around so we need to transpose the matrix first
        cov = np.cov(X.T)

        #eigenvectors, values
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        #in the documentation eigenvalues returned as column vectors as v[:, i] = column i is the eigenvector
        #sort first eigenvectors
        eigenvectors = eigenvectors.T
        #we want to have it in decreasing order so we do the slicing to reverse a list
        idxs = np.argsort(eigenvalues)[::-1]
        #now we have the indices of the sorted eigenvalues in the decreasing order
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        #store first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self,X):
        #we need to project the data and subtract the mean
        X = X - self.mean
        #now we can project it and then return it/ Project X onto our components
        #we transposed them previously and now we need a column vector so we transpose them again
        return np.dot(X, self.components.T)


#now we can test our algo
from sklearn import datasets

data = datasets.load_iris()
X = data.data
y = data.target

pca = PCA(2)
pca.fit(X)
x_projected = pca.transform(X)

print('Shape of X:'), X.shape
print('Shape of transfornmed X:', X_projected.shape)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

#lets plot our stuff
import matplotlib.pyplot as plt

plt.scatter(x1, x2,
            c = y, edgecolor = 'none', alpha = 0.8,
            cmap = plt.cm.get_cmap('viridis', 3))

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.colorbar()
plt.show()
