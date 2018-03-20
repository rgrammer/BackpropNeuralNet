from __future__ import print_function

#standard imports
import math
import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
import random as rn
from copy import deepcopy

#sklearn imports
from sklearn.cross_validation import cross_val_predict
from sklearn.preprocessing import MinMaxScalar

#nn imports
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split


def queryCsv():
    df = pd.read_csv('data/dataset.csv')
    print(df.shape)
    print(list(df.columns.values))
    return df

#load datafram into 2D array, create sets of distinct values for each column, and get data types for each of the columns
def loadData(df):
    distinctValues = []
    colTypes = ["categorical", "numeric"]
    for col in df:
        distinctValues.append(df[col].unique())
    return distinctValues, colTypes

#def calculateMean for numeric columns
def getMeans(df, colTypes):
    result = []
    for col in range(len(colTypes)):
        sum = 0.0
        if colTypes[col] == "numeric":
            for index, row in df.iterrows():
                val = pd.to_numeric(df.iloc[index, col])
                sum += val
            result.append(sum/df.shape[0])
    return result

#determine std devs for numeric columns
def getStdDev(df, colTypes, means):
    result = []
    meansCol = 0
    for col in range(len(colTypes)):
        sum = 0.0
        if colTypes[col] == "numeric":
            for index, row in df.iterrows():
                val = pd.to_numeric(df.iloc[index, col])
                sum = (val - means[meansCol]) * (val - means[meansCol])
            result.append(math.sqrt(sum / df.shape[0]))
            meansCol += 1
    return result

#sigmoid activation function
def sigmoid(x, deriv=False):
    if(deriv):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

#linear activation function
def linear(x, deriv=False):
    if(deriv):
        return 1
    return x

#tanh activation function
def tanh(x):
    return 1 - (x**2)

########conversion methods#########
def binaryIndepenToValue(val, col, distinctValues):
    if(len(distinctValues[col]) != 2):
        print("Binary X data only 2 values allowed")
        exit
    if(distinctValues[col][0] == val):
        return -1.0
    else:
        return 1.0

def binaryDepenToValue(val, col, distinctValues):
    if(len(distinctValues[col]) != 2):
        print("Binary Y data only 2 values allowed")
        exit
    if(distinctValues[col][0] == val):
        return 0.0
    else
        return 1.0

def catIndependentToValues(val, col, distinctValues):
    if(len(distinctValues[col]) == 2):
        print("Categorical X data only 1, 3+ values allowed")
    size = len(distinctValues[col])
    result = [0] * size
    idx = 0
    for i in range(size):
        if distinctValues[col][i] == val:
            idx = i
            break
    if idx == size - 1: #last val all -1.0s
        for i in range(size): #ex: [-1.0, -1.0, -1.0]
            result[i] == -1.0
    else:
        result[len(result) - 1 - idx] = 1.0
    return result

def catDepenToValues(val, col, distinctValues):
    if(len(distinctValues[col]) == 2):
        print("Categorical X data only 1, 3+ values allowed")
    size = len(distinctValues[col])
    result = [size]
    idx = 0

    for i in range(size):
        if distinctValues[col][i] == val:
            idx = i
            break
    result[len(result) - 1 - idx] = 1.0 #ex: [0.0, 1.0, 0.0]
    return result

def numIndependenToValue(val, col, means, stdDevs):
    x = val
    m = means[col]
    sd = stdDevs[col]
    return (x - m)/sd

def numNewCols(distinctValues, colTypes):
    result = 0
    for i in range(len(colTypes)):
        if colTypes[i] == "categorical":
            numCatValues = len(distinctValues[i])
            result = result + numCatValues + 1
    return result

#transforms the raw data to normalized and encoded data
def transform(df):
    distinctValues, colTypes = loadData(df)
    extraCols = numNewCols(distinctValues, colTypes)
    print(extraCols)
    result = [[0 for x in range(extraCols)] for y in range(df.shape[0])]
    means = getMeans(df, colTypes)
    stdDevs = getStdDev(df, colTypes, means)
    for row in range(df.shape[0]):
        k = 0
        for col in range(df.shape[1]):
            val = df.iloc[row,col]
            isBinary = (colTypes[col] == "binary")
            isCategorical = (colTypes[col] == "categorical")
            isNumeric = (colTypes[col] == "numeric")
            isIndependent = (col < df.shape[1] - 1)
            isDependent = (col == df.shape[1] - 1)
            if(isBinary and isIndependent): #binary x value -> -1.0 or +1.0
                result[row][k] = binaryIndepenToValue(val, col, distinctValues)
                k += 1
            elif(isBinary and isDependent):
                result[row][k] = binaryDepenToValue(val, col, distinctValues)
            elif(isCategorical and isIndependent):
                vals = catIndependentToValues(val, col, distinctValues)
                for j in range(len(vals)):
                    result[row][k] = vals[j]
                    k += 1
            elif (isCategorical and isDependent):
                vals = catDepenToValues(val, col, distinctValues)
                for j in range(len(vals)):
                    result[row][k] = vals[j]
                    k += 1
            elif(isNumeric and isIndependent):
                result[row][k] = numIndependenToValue(val, col, means, stdDevs)
                k += 1
            elif(isNumeric and isDependent):
                result[row][k] = val
    return result

#backprop algorithm
def backprop(X, y):
    np.random.seed(1)

    inputs = len(X[0])
    hiddenLayers = 10
    outputs = 1

    #randomly initialize our weights with mean 0
    syn0 = 2 * np.random.random((inputs, hiddenLayers)) - 1
    syn1 = 2 * np.random.random((hiddenLayers, outputs)) - 1

    for j in range(100000):
        #feed forward through layers 0, 1, and 2
        l0 = X
        l1 = sigmoid(np.dot(l0, syn0))
        l2 = sigmoid(np.dot(l1, syn1))

        #how much did we miss the target value?
        l2_error = y - l2
        if(j % 10000) == 0:
            print("Error:" + str(np.mean(np.abs(l2_error))))

        #in what direction is the target value? and how far were we?
        l2_delta = l2_error * sigmoid(l2, deriv=True)
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)
        #in what direction is the target l1?
        #were we really sure? if so, dont change too much
        l1_delta = l1_error * sigmoid(l1, deriv=True)
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

    print("Error:" + str(np.mean(np.abs(l2_error))))

    return syn0, syn1

def test(syn0, syn1, part):
    l1 = sigmoid(np.dot(part, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    return l2

if __name__ == '__main__':
    df = queryCsv()
    data = transform(df)
    X = np.array(data)[:,:-1]
    y = np.reshape(([row[-1] for row in data]), (-1, 1))
    maxVal = max(y)
    scaler = MinMaxScalar()

    scaler.fit(y)
    y = scaler.transform(y)
    print(y)
    syn0, syn1 = backprop(X, y)
    result = test(syn0, syn1, X)
    result = maxVal * result
    print(result)

