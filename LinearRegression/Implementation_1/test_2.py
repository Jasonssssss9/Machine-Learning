from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from LinearRegression import LinearRegression


# 加载糖尿病数据集
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 使用sklearn中的递归下降算法进行模型训练
model = SGDRegressor(eta0=0.01, max_iter=10000, tol=1e-3)
model.fit(X_train, y_train)
predictions_1 = model.predict(X_test)

# 计算均方误差
mse_1 = mean_squared_error(y_test, predictions_1)
print("Sklearn mse:", mse_1)


# 使用LinearRegression的方法进行模型训练
linear_regression = LinearRegression(X_train, y_train)
(theta, mse_history) = linear_regression.train(0.01, 1000)

# 打印一个mes_history
plt.plot(mse_history)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('Iteration-MSE diagram')
plt.show()
print("My initial mse:", mse_history[0])
print("My final mse:", mse_history[-1])

