# 计算7种的比例 7到7
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import  pandas as pd
from keras.models import Sequential, load_model
import csv
from sklearn import preprocessing

dataframe = pd.read_csv('E:/math/题目/C/sevenall.csv', engine='python')#导入数据
# print(data)
data = dataframe.values
# 将整型变为float
data = data.astype('float32')

# 将数据进行归一化处理，将值缩放到 [0, 1] 的范围内
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

#必须先用fit_transform(trainData)，之后再transform(testData)
# 如果直接transform(testData)，程序会报错
# 如果fit_transfrom(trainData)后，使用fit_transform(testData)而不transform(testData)，虽然也能归一化，但是两个结果不是在同一个“标准”下的，具有明显差异。(一定要避免这种情况
#fit 简单来说，就是求得训练集X的均值啊，方差啊，最大值啊，最小值啊这些训练集X固有的属性。可以理解为一个训练过程
# transform 在Fit的基础上，进行标准化，降维，归一化等操作（看具体用的是哪个工具，如PCA，StandardScaler等


# 准备训练数据和目标数据
look_back = 5
trainX, trainY = [], []
for i in range(len(data)-look_back):
    trainX.append(data[i:i+look_back, :])
    trainY.append(data[i+look_back, :])
trainX = np.array(trainX)
trainY = np.array(trainY)

#训练数据太少 look_back并不能过大
# 训练模型过程中随时都要注意目标函数值 (loss)的大小变化。 一个正常的模型loss应该随训练轮数（epoch）的增加而缓慢下降，
# 然后趋于稳定。 虽然在模型训练的初始阶段，loss有可能会出现大幅度震荡变化，但是只要数据量充分，模型正确，
# 训练的轮数足够长，模型最终会达到收敛状态，接近最优值或者找到了某个局部最优值。

# 创建 LSTM 模型
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 7)))
model.add(Dense(7))
model.compile(loss='mean_squared_error', optimizer='adam')


# 训练模型
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)



# testPredict=multi_step_prediction(model,trainX,30)

testX = np.array([data[-look_back:, :]])

#单次
# testPredict = model.predict(testX)
# testPredict = scaler.inverse_transform(testPredict)

#多次
predictions = []
for i in range(60):
    # 预测下一个时间步的数据
    yhat = model.predict(testX)
    # 将预测结果添加到 predictions 中
    predictions.append(yhat[0])
    # 更新输入数据，以便用于下一个时间步的预测
    testX = np.concatenate([testX[:, 1:, :], np.reshape(yhat, (1, 1, -1))], axis=1)
predictions = np.array(predictions)

# 将预测结果进行反归一化处理，得到原始的数据值

predictions = scaler.inverse_transform(predictions) 


with open("E:/math/题目/C/ss.csv","w") as f:
    b_csv=csv.writer(f)
    b_csv.writerows(predictions)  #批量写入
# print(predictions)

