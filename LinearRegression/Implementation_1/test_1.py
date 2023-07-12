import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from LinearRegression import LinearRegression

# 获取世界幸福指数数据
data = pd.read_csv('world-happiness-report-2017.csv')

train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)

features_name = 'Economy..GDP.per.Capita.'
target_name = 'Happiness.Score'

X_train = train_data[[features_name]].values
y_train = train_data[target_name].values
X_test = test_data[[features_name]].values
y_test = test_data[target_name].values

linear_regression = LinearRegression(X_train,y_train)
(theta,mse_history) = linear_regression.train(0.01, 1000)

print ('My initial mse:',mse_history[0])
print ('My final mse:',mse_history[-1])

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