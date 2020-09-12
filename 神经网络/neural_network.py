import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
y = [0, 1, 1, 0, 0]
x0 = [1, 1, 1, 1, 1]
x1 = [0, 1, 1, 0, 0]
x2 = [0, 1, 0, 1, 1]
x3 = [1, 1, 1, 1, 1]


def sigmod_func(x):
    # 激活函数
    return 1 / (1 + np.exp(-x))

def hidden_layer(x, theta=[]):
    # 构建隐藏层
    # theta参数矩阵4*4（4对应输入层变量个数，4对应隐藏层神经元个数）
    # 生成z值矩阵
    z = x*theta
    # 传入激活函数，生成a值矩阵
    a = sigmod_func(z)
    return a, theta
def out_layer(a, theta=[]):
    # 构建输出层
    # theta参数矩阵5*1（5对应输入层变量个数包括了偏置项，1对应隐藏层神经元个数）
    # 输出矩阵
    z = a*theta
    # 传入激活函数，生成a值矩阵
    out = sigmod_func(z)
    return out, theta
def randInitializeWeights(L_in, L_out):
    theta = np.random.random(size=(L_in, L_out))
    theta = (theta - theta.mean()) / (theta.max() - theta.min())
    return theta
def computeCost(y, y_hat, theta, lamb):
    # 计算代价函数
    inner = np.multiply(y, np.log(y_hat)) + np.multiply((1 - y), np.log(1 - y_hat))
    reg_sum = 0
    for t in theta:
        # 去除偏置项的参数t[1:, 0], 第一行为偏置项的参数
        reg = np.sum(np.power(t[1:, :], 2)) * lamb / (2 * len(y))
        reg_sum += reg
    return np.sum(inner) / (-1*len(y)) + reg_sum


def neuralNetwork(x, y, alpha= 0.01, lamb= 1, iterations=1500):
    cost_list = []
    for i in range(iterations):  # 开始迭代
        m = len(y)  # 样本个数
        # 前向传播
        if i == 0:
            # 增加偏置项
            bias = np.mat(np.ones(x.shape[0])).T
            x = np.hstack([bias, x])
            theta1 = randInitializeWeights(x.shape[1], 4)  # 初始化隐藏层的权重矩阵，4表示隐含层有4个神经单元（不包括偏置项）
            a, theta1 = hidden_layer(x, theta1)  # 计算隐藏层
            # 增加偏置项
            bias = np.mat(np.ones(a.shape[0])).T
            a = np.hstack([bias, a])
            theta2 = randInitializeWeights(a.shape[1], 1)  # 初始化输出层的权重矩阵，1表示二元分类
            out, theta2 = out_layer(a, theta2)  # 计算输出层
        else:
            [theta1, theta2] = theta
            a, theta1 = hidden_layer(x, theta1)  # 计算隐藏层
            # 增加偏置项
            bias = np.mat(np.ones(a.shape[0])).T
            a = np.hstack([bias, a])
            out, theta2 = out_layer(a, theta2)  # 计算输出层
        theta = [theta1, theta2]
        # 计算代价函数
        cost = computeCost(y, out, theta, lamb=lamb)
        if cost == math.nan:
            print("无法收敛")
        cost_list.append(cost)
        # 反向传播
        error2 = out - y  # 总误差
        error1 = np.multiply(np.multiply(a, 1 - a), error2 * theta2.T)  # 隐藏层误差
        error1 = error1[:, 1:]  # 去除偏置项
        step2 = alpha / m * np.multiply(a, error2).sum(axis=0)
        theta2[0, :] -= step2[0, 0]
        theta2[1:, :] = (1 - alpha * lamb / m) * theta2[1:, :] - step2[0, 1:].T
        step1 = alpha / m * np.multiply(x, error1).sum(axis=0)
        theta1[0, :] -= step1[0, 0]
        theta1[1:, :] = (1 - alpha * lamb / m) * theta1[1:, :] - step1[0, 1:].T
        theta = [theta1, theta2]
    return theta, cost_list, out

# 构建三层神经网络,二元分类(1 - alpha * lamb / m) * theta2[1:, :]
# 构建输入层n*4
x = np.mat(np.vstack([x1, x2, x3]).T)
y = np.mat(y).T

theta, error_list, out = neuralNetwork(x, y, alpha=0.1, lamb=0.1, iterations=1500)
# print(error2)
out[out >= 0.5] = 1
out[out < 0.5] = 0

y_hat = pd.DataFrame(out)
y = pd.DataFrame(y)
print(error_list[-1])

# 拟合图
df = pd.concat([y, y_hat], axis=1)
df.index = x1
df.columns = ["y", "y_hat"]
df.index = range(0, len(df.index))
df.plot()
plt.show()
print(df)

cost_df = pd.DataFrame(error_list, index=range(len(error_list)))
cost_df.columns = ["cost"]
cost_df = cost_df.sort_index(ascending=False)
# ddf.set_index("theta")
cost_df.plot()
plt.show()