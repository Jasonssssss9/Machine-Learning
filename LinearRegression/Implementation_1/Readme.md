# 机器学习线性回归递归下降算法实现



## 一、基本原理

线性回归是一种常见的**有监督机器学习算法**，用于预测一个连续值的输出。梯度下降是一种优化算法，用于找到能够最小化目标函数的参数值。在线性回归中，梯度下降被用来最小化成本函数，最小化预测值与实际值之间的误差。

<img src=".\image\1.jpg" alt="1" style="zoom:30%;" />

<img src=".\image\2.jpg" alt="2" style="zoom:30%;" />



## 二、numpy代码实现

本实验以python numpy库为基础，借助sklearn.preprocessing进行数据预处理

#### 1.训练和预测模型

在LinearRegression.py文件中定义LinearRegression类，实现梯度下降算法

* **初始化**

  ~~~python
  class LinearRegression:
      '''
      一个线性回归机器学习模型
      主要分为两个功能：(1)传入训练集进行模型训练, 得到θ参数; (2)传入测试集进行预测
      '''
  
      def  __init__(self, features, targets):
          # 预处理特征值:
          features  = self.__preprocessing(features)
  
          self.features = features #特征值
          self.targets = targets   #目标值
          self.theta = np.zeros(self.features.shape[1]) #初始化θ值
  ~~~

  主要设置features(特征值)，targets(目标值)和theta(参数)为成员变量，首先需要调用__proprocessing函数对features对特征值进行预处理

* **预处理**

  ~~~python
  def __preprocessing(self, data):
      '''
      预处理数据
      '''
      scaler = StandardScaler()
      data = scaler.fit_transform(data)
      # 给features增加一列1
      num_rows = data.shape[0]
      ones_col = np.ones((num_rows,1))
      data = np.hstack((data, ones_col))
      return data
  ~~~

  使用sklearn提供的StandardScaler进行数据标准化，即features先减去均值再除以标准差，**让数据分布在原点周围，并且具有单位标准差**

  之后再**给features增加一列1**
  因为最终的线性回归方程为y=aX+b，**增加一列1保证可以获得常数项**

  > 注意，这里很容易忘记常数项

* **训练模型**

  * train函数

    ~~~python
    def train(self, alpha = 0.01, num_iterations = 500):
        '''
        传入学习率和迭代次数参数，进行模型训练
        返回最终得到的theta，以及迭代过程中每一步的均方误差作为测试信息
        '''
        mse_history = self.__gradient_descent(alpha, num_iterations)
        return self.theta, mse_history
    ~~~

    train为暴露给外部的接口，实现模型训练，需要传入**学习率(步长)α**和**最大迭代次数**

    返回参数theta，以及一个mse_history列表

    这个mse_history列表存储迭代中每一步的均方误差

  * __gradient_descent函数

    ~~~python
    def __gradient_descent(self, alpha, num_iterations):
        '''
        实际进行梯度下降算法
        生成最终的θ
        返回迭代中每一步的均方误差，用来在测试中显示迭代过程
        '''
        mse_history = []
        for _ in range(num_iterations):
            self.__gradient_step(alpha) #更新θ
            mse_history.append(self.__current_MSE())
        return mse_history
    ~~~

    进行梯度下降算法，迭代num_iterations次，每一步调用`__gradient_step`函数更新theta，再调用`__current_MSE`函数获取当前均方误差，加入到mes_history中

  * __gradient_step和\_\_curr_prediction函数

    ~~~python
    def __curr_prediction(self, data, theta):
        '''
        计算预测结果
        '''
        return np.dot(data, theta)
    
    def __gradient_step(self, alpha):
        '''
        实现每一步计算θ的公式
        '''
        curr_prediction = self.__curr_prediction(self.features, self.theta)
        delta = curr_prediction - self.targets
        self.theta = self.theta - alpha*(1/self.features.shape[0])*(np.dot(delta.T, self.features)).T
    ~~~

    以上即矩阵实现梯度下降法的公式

  * __current_MSE函数

    ~~~python
    def __current_MSE(self):
        '''
        计算当前模型的均方误差(Mean Squared Error)
        '''
        curr_prediction = self.__curr_prediction(self.features, self.theta)
        delta = curr_prediction - self.targets
        mse = (1/2)*np.dot(delta.T, delta)/self.features.shape[0]
        return mse
    ~~~

    计算当前均方误差

* 使用模型进行预测

  ~~~python
  def predict(self, data):
      '''
      根据训练得到的模型进行数据预测
      '''
      #预处理数据：
      data = self.__preprocessing(data)
  
      return self.__curr_prediction(data, self.theta)
  ~~~

  传入需要预测的数据(测试集数据)，返回预测结果

####  2.测试1

~~~python
# test_1.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from LinearRegression import LinearRegression

# 获取世界幸福指数数据
data = pd.read_csv('world-happiness-report-2017.csv')

train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)

# 为了方便打印结果，这里特征值只选取一列数据
features_name = 'Economy..GDP.per.Capita.'
target_name = 'Happiness.Score'

# 获取特征值和目标值
X_train = train_data[[features_name]].values
# 注意这里要用两个[]，因为即使只有一列数据，X也应该是一个二维数组
y_train = train_data[target_name].values
X_test = test_data[[features_name]].values
y_test = test_data[target_name].values

# 进行模型训练
linear_regression = LinearRegression(X_train,y_train)
(theta,mse_history) = linear_regression.train(0.01, 1000)

# 打印初始和最终均方误差
print ('My initial mse:',mse_history[0])
print ('My final mse:',mse_history[-1])

# 打印均方误差和迭代次数图像
plt.plot(mse_history)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('Iteration-MSE Diagram')
plt.show()

predictions_num = 100
x_predictions = np.linspace(X_train.min(),X_train.max(),predictions_num).reshape(predictions_num,1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(X_train,y_train,label='Train data')
plt.scatter(X_test,y_test,label='Test data')
plt.plot(x_predictions,y_predictions,'r',label = 'Prediction')
plt.xlabel(features_name)
plt.ylabel(target_name)
plt.title('Linear Regression Diagram')
plt.legend()
plt.show()
~~~

步骤：

①获取数据；②对数据进行划分；③进行模型训练；④打印均方误差查看训练效果；⑤进行数据预测；⑥打印数据预测结果

> 注意：这里实际上没有用到测试集，直接通过X_train.min()和X_train_max()生成一个新的100个测试数据

结果：

MSE迭代减小图像：

<img src=".\image\3.jpg" alt="3" style="zoom:50%;" />

拟合结果

<img src=".\image\4.jpg" alt="4" style="zoom:50%;" />

打印结果

![5](.\image\5.jpg)

#### 3.测试2

测试1的数据较为简单，并且只选取了特征值的一列。测试2借助sklearn提供的糖尿病数据，将本代码和sklearn提供的线性回归递归下降算法的预测结果进行比较

~~~python
# test_2.py

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

predictions_2 = linear_regression.predict(X_test)
mse_2 = mean_squared_error(y_test, predictions_2)
print("My mse:", mse_2)
~~~

<img src=".\image\6.jpg" alt="6" style="zoom:50%;" />

![7](.\image\7.jpg)

可以看到最终结果差异不大













