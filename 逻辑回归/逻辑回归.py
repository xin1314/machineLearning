import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
def sigmod_func(z):
    return 1 / (1 + np.exp(-z))

num_list = np.linspace(-10, 10, 100)
num_list = sorted(num_list)
chg_num = [sigmod_func(x) for x in num_list]
df = pd.DataFrame(chg_num, index=num_list)
df.plot()
plt.show()

def computeCost(x, y, theta):
    inner = np.multiply(-y, np.log(1 + np.exp(-x * theta.T))) - np.multiply((1 - y), np.log(1 + np.exp(x * theta.T)))
    return np.sum(inner) / (-1*len(x))




# 梯度下降
def sig_gradient_descent_func(data, theta, alpha=0.001, iterations=1500):
    '''
    逻辑回归
    :param data:
    :param theta:
    :param alpha:
    :param iterations:
    :return:
    '''
    x = np.mat(data.iloc[:, :-1])
    y = np.mat(data.iloc[:, -1]).T
    temp_theta = np.matrix(np.zeros(theta.shape))

    theta_list = np.zeros((iterations, x.shape[1]))  # 保存每次迭代的theta值
    cost_list = np.zeros(iterations)  # 保存每次迭代的损失函数值
    m = x.shape[0]  # 样本个数
    colNum = x.shape[1]  # 变量个数
    for i in range(iterations):  # 开始迭代
        for j in range(colNum):  # 循环多变量
            step = alpha / m * np.sum(np.multiply(1 / (1 + np.exp(-x * theta.T)) - y, x[:, j]))
            temp_theta[0, j] = theta[0, j] - step  # 临时变量，保证theta在一次迭代中值不变
            theta_list[i, j] = theta[0, j] - step  # 保存每次迭代的theta值
        theta = temp_theta  # 更新参数
        cost_list[i] = computeCost(x, y, theta)  # 保存损失函数值
    return theta_list, cost_list


y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
x1 = [1, 1.1, 2.1, 2.2, 3.2, 4.3, 5, 7.5, 8.2, 9.3]

data = np.vstack((x1, y))
data = pd.DataFrame(data).T
# 特征缩放
scaled_data = (data - data.mean()) / (data.max() -data.min())
x0 = pd.DataFrame([1]*len(x1))
data = pd.concat([x0, data], axis=1)
theta = np.matrix(np.array([0, 0]))

iterations = 25000
alpha = 0.01
# alpha = 0.01 0.03 0.1 0.3 1 3 10
theta_list, cost_list = sig_gradient_descent_func(data, theta, alpha=alpha, iterations=iterations)
plt.plot(cost_list)
plt.show()  # 显示图像
cost_list = list(cost_list)
cost_list = list(cost_list)
best_theta = theta_list[cost_list.index(min(cost_list))]
print(best_theta)
print("min cost_list:", min(cost_list))
y_hat = 1 / (1 + np.exp(-np.sum(data.iloc[:, :-1] * best_theta, axis=1)))
y_hat[y_hat >= 0.5] = 1
y_hat[y_hat < 0.5] = 0
# 拟合图
df = pd.concat([data.iloc[:, -1], y_hat], axis=1)
df.index = x1
df.columns = ["y", "y_hat"]
df.plot()
plt.show()
print(df)

ddf = pd.DataFrame(cost_list, index=theta_list[:, 1])
ddf.columns = ["cost"]
ddf = ddf.sort_index(ascending=False)
# ddf.set_index("theta")
ddf.plot()
plt.show()



