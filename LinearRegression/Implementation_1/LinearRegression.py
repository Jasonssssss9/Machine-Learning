import numpy as np
from sklearn.preprocessing import StandardScaler


class LinearRegression:
    '''
    一个线性回归机器学习模型
    主要分为两个功能：(1)传入训练集进行模型训练, 得到θ参数; (2)传入测试集进行预测
    '''

    def  __init__(self, features, targets):
        # 预处理特征值:
        self.__preprocessing(features)

        self.features = features #特征值
        self.targets = targets   #目标值
        self.theta = np.zeros(self.features.shape[1]) #初始化θ值

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
        
    def train(self, alpha = 0.01, num_iterations = 500):
        '''
        传入学习率和迭代次数参数，进行模型训练
        返回最终得到的theta，以及迭代过程中每一步的均方误差作为测试信息
        '''
        mse_history = self.__gradient_descent(alpha, num_iterations)
        return self.theta, mse_history
    
    def predict(self, data):
        '''
        根据训练得到的模型进行数据预测
        '''
        #预处理数据：
        self.__preprocessing(data)

        return self.__curr_prediction(data, self.theta)
    
    def __curr_prediction(self, data, theta):
        '''
        计算预测结果
        '''
        return np.dot(data, theta)
        

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

    def __gradient_step(self, alpha):
        '''
        实现每一步计算θ的公式
        '''
        curr_prediction = self.__curr_prediction(self.features, self.theta)
        delta = curr_prediction - self.targets
        self.theta = self.theta - alpha*(1/self.features.shape[0])*(np.dot(delta.T, self.features)).T

    def __current_MSE(self):
        '''
        计算当前模型的均方误差(Mean Squared Error)
        '''
        curr_prediction = self.__curr_prediction(self.features, self.theta)
        delta = curr_prediction - self.targets
        mse = (1/2)*np.dot(delta.T, delta)/self.features.shape[0]
        return mse
