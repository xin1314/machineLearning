# 只适用于线性模型，不适合逻辑回归模型等其他模型
# 需要计算（XTX）-1 如果特征数量n较大则运算代价大，因为矩阵逆的计算时间复杂度为，通常来说当小于10000 时还是可以接受的
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
y = [1, 2.2, 3.5, 4, 5.4, 4.6, 7, 6.8, 9, 10.2]
x1 = [1, 1.1, 2.1, 2.2, 3.2, 4.3, 5, 7.5, 8.2, 9.3]
x2 = np.array(x1) ** 2
x3 = np.array(x1) ** 3
data = np.vstack((x1, x2, x3, y))
data = pd.DataFrame(data).T

def normalEqn(data):
    x0 = np.mat([1]*data.shape[0]).T
    x1 = np.mat(data.iloc[:, :-1])
    x = np.hstack((x0, x1))
    y = np.mat(data.iloc[:, -1]).T
   # theta = (np.linalg.pinv(x.T.dot(x)).dot(x.T)).dot(y)
    theta = (np.linalg.pinv(x.T*x)*x.T)*y

    cost_value = computeCost(x, y, theta.T)
    return theta, cost_value

# 损失函数
def computeCost(x, y, theta):
    inner = np.power(x * theta.T - y, 2)
    return np.sum(inner) / (2*len(x))

best_theta, cost_value = normalEqn(data)
print(best_theta)

print("min cost_value:", cost_value)
x0 = np.mat([1]*data.shape[0]).T
x1 = np.mat(data.iloc[:, :-1])
x = np.hstack((x0, x1))

y_hat = pd.DataFrame(np.sum(x * best_theta, axis=1))
# 拟合图
df = pd.concat([data.iloc[:, -1], y_hat], axis=1)
df.index = data.iloc[:, 0]
df.columns = ["y", "y_hat"]
df.plot()
plt.show()
print(df)












