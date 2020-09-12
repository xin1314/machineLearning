y = [1,2,3,4,5,6,7,8,9,10]
x1 = [1,1.1,2.1,1.2,3.2,4.3,5,7.5,3.2,8.3]


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
plt.plot(x1, y)
plt.show()  # 显示图像

# 损失函数
def computeCost(x, y, theta):
    inner = np.power(x * theta.T - y, 2)
    return np.sum(inner) / (2*len(x))




# 梯度下降
def gradient_descent_func(data, theta, alpha=0.001, iterations=1500):
    x = np.mat(data.iloc[:, :-1])
    y = np.mat(data.iloc[:, -1]).T
    temp_theta = np.matrix(np.zeros(theta.shape))

    theta_list = np.zeros((iterations, x.shape[1]))  # 保存每次迭代的theta值
    cost_list = np.zeros(iterations)  # 保存每次迭代的损失函数值
    m = x.shape[0]  # 样本个数
    colNum = x.shape[1]  # 变量个数
    for i in range(iterations):  # 开始迭代
        for j in range(colNum):  # 循环多变量
            step = alpha / m * np.sum(np.multiply(x * theta.T - y, x[:, j]))
            temp_theta[0, j] = theta[0, j] - step  # 临时变量，保证theta在一次迭代中值不变
            theta_list[i, j] = theta[0, j] - step  # 保存每次迭代的theta值
        theta = temp_theta  # 更新参数
        cost_list[i] = computeCost(x, y, theta)  # 保存损失函数值
    return theta_list, cost_list


y = [1, 2.2, 3.5, 4, 5.4, 4.6, 7, 6.8, 9, 10.2]
x0 = pd.DataFrame([1]*10)
x1 = [1, 1.1, 2.1, 2.2, 3.2, 4.3, 5, 7.5, 8.2, 9.3]
x2 = np.array(x1) ** 2
x3 = np.array(x1) ** 3
x4 = np.array(x1) ** 4
data = np.vstack((x1, x2, x3, x4, y))
data = pd.DataFrame(data).T
# 特征缩放
data = (data - data.mean()) / (data.max() -data.min())
data = pd.concat([x0, data], axis=1)
theta = np.matrix(np.array([0, 0, 0,0,0]))

iterations = 25000
alpha = 0.3
# alpha = 0.01 0.03 0.1 0.3 1 3 10
theta_list, cost_list = gradient_descent_func(data, theta, alpha=alpha, iterations=iterations)
plt.plot(cost_list)
plt.show()  # 显示图像
cost_list = list(cost_list)
cost_list = list(cost_list)
best_theta = theta_list[cost_list.index(min(cost_list))]
print(best_theta)
print("min cost_list:", min(cost_list))
y_hat = np.sum(data.iloc[:, :-1] * best_theta, axis=1)
# 拟合图
df = pd.concat([data.iloc[:, -1], y_hat], axis=1)
df.index = x1
df.columns = ["y", "y_hat"]
df.plot()
plt.show()
print(df)


# ddf = pd.DataFrame(cost_list, index=theta_list[:, 1])
# ddf.columns = ["cost"]
# ddf = ddf.sort_index(ascending=False)
# # ddf.set_index("theta")
# ddf.plot()
# plt.show()