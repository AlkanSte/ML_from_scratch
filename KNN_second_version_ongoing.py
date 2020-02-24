import pandas as pd
import numpy as np
import math
import operator
from sklearn import datasets

#iris = datasets.load_iris()
#X, y = iris.data, iris.target

def euclideanDistance(data1,data2,length):
    distance=0
    for x in range(length):
        distance+=np.square(data1[x]-data2[x])
    return np.sqrt(distance)

# Defining our KNN model
def knn(train,test,k):
    
    
    length=test.shape[1]
    result=[]
    # calculating euclideanDistance between each row of training data and test data
    for y in range(len(test)):
        distances={}
        sort={}
        for x in range(len(train)):
            dist=euclideanDistance(test.iloc[y],train.iloc[x],length)
            distances[x]=dist
           
        #sorting them on the basis of distance
        sorted_d=sorted(distances.items(),key=operator.itemgetter(1))
    
        neighbors=[]
    
        # Extracting top k neighbors
        for  x in range(k):
            neighbors.append(sorted_d[x][0])
        
        classvotes={}
    
        # calculate most frequent class in the neighbors
        for x in range(len(neighbors)):
            response=train.iloc[neighbors[x]][-1]
        
            if response in classvotes:
                classvotes[response]+=1
            else:
                classvotes[response]=1
         
        sortedvotes=sorted(classvotes.items(),key=operator.itemgetter(1),reverse=True)
        result.append(sortedvotes[0][0])
    return (result)
    
#Ongoing
