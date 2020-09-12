# 多变量梯度下降算法的实现，数据集采用吴恩达机器学习教程“ex1data2.txt”
# 对于多变量线性回归梯度下降算法的实现，这里采用向量化的方式去进行

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def readData(path, name=[]):
    data = pd.read_csv(path, names=name)
    data = (data - data.mean()) / data.std()
    data.insert(0, 'First', 1)
    return data


def costFunction(X, Y, theta):
    inner = np.power(((X * theta.T) - Y.T), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(data, theta, alpha, iterations):
    eachIterationValue = np.zeros((iterations, 1))
    temp = np.matrix(np.zeros(theta.shape))
    X = np.matrix(data.iloc[:, 0:-1].values)
    print(X)
    Y = np.matrix(data.iloc[:, -1].values)
    m = X.shape[0]
    colNum = X.shape[1]
    for i in range(iterations):
        error = (X * theta.T) - Y.T
        for j in range(colNum):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / m) * np.sum(term))
        theta = temp
        eachIterationValue[i, 0] = costFunction(X, Y, theta)
    return theta, eachIterationValue


if __name__ == "__main__":
  #  data = readData('ex1data2.txt', ['Size', 'Bedrooms', 'Price'])
    # data = (data - data.mean()) / data.std()
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x1 = [1, 1.1, 2.1, 1.2, 3.2, 4.3, 5, 7.5, 3.2, 8.3]
    x2 = np.array(x1) ** 2
    x3 = np.array(x1) ** 3
    x = np.vstack((x1, x2, x3, y))
    data = pd.DataFrame(x).T
    theta = np.matrix(np.array([0, 0, 0]))

    iterations = 1500
    alpha = 0.000001

    theta, eachIterationValue = gradientDescent(data, theta, alpha, iterations)

    print(theta)

    plt.plot(np.arange(iterations), eachIterationValue)
    plt.title('CostFunction')
    plt.show()
