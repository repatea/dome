#分数的预测

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import  pandas as pd
import  os
from keras.models import Sequential, load_model
import csv
from sklearn import preprocessing


dataframe = pd.read_csv('E:/math/题目/text/numberof.csv', engine='python')#导入数据
dataset = dataframe.values
# 将整型变为float
dataset = dataset.astype('float32')

#归一化 
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#必须先用fit_transform(trainData)，之后再transform(testData)
# 如果直接transform(testData)，程序会报错
# 如果fit_transfrom(trainData)后，使用fit_transform(testData)而不transform(testData)，虽然也能归一化，但是两个结果不是在同一个“标准”下的，具有明显差异。(一定要避免这种情况
#fit 简单来说，就是求得训练集X的均值啊，方差啊，最大值啊，最小值啊这些训练集X固有的属性。可以理解为一个训练过程
# transform 在Fit的基础上，进行标准化，降维，归一化等操作（看具体用的是哪个工具，如PCA，StandardScaler等

#拆分
train_size = int(len(dataset) * 1)
trainlist = dataset[:train_size]
testlist = dataset[train_size:]

#timestep=2步预测后面的数据1，2->3;2,3->4
def create_dataset(dataset, look_back):
#这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]#很多个串联起来
        dataX.append(a)
        dataY.append(dataset[i + look_back])#预测值或者说目标值
    return np.array(dataX),np.array(dataY)

look_back = 1
trainX,trainY  = create_dataset(trainlist,look_back)
# testX,testY = create_dataset(testlist,look_back)

#模型部分
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

# 创建一个 Sequential 模型
model = Sequential()
model.add(LSTM(4, input_shape=(None,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# 使用训练好的模型对训练集和测试集进行预测
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)

testX = np.array([trainY[-look_back:]])

#单次
# testPredict = model.predict(testX)
# testPredict = scaler.inverse_transform(testPredict)

#多次
predictions = []
for i in range(61):
    # 预测下一个时间步的数据
    yhat = model.predict(testX)
    # 将预测结果添加到 predictions 中
    print(yhat)
    predictions.append(yhat[0])
    # 更新输入数据，以便用于下一个时间步的预测
    testX = np.concatenate([testX[:, 1:, :], np.reshape(yhat, (1, 1, -1))], axis=1)
predictions = np.array(predictions)



#反归一化
trainPredict = scaler.inverse_transform(predictions)
trainY = scaler.inverse_transform(trainY)
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform(testY)
x=range(len(trainY),len(trainY)+len(trainPredict)-1)

with open("E:/math/题目/C/ee.csv","w") as f:
    b_csv=csv.writer(f)
    b_csv.writerows(trainPredict)  #批量写入

print(len(trainPredict[1:]))
plt.plot(trainY)
plt.plot(x,trainPredict[1:])
plt.show()
# plt.plot(testY)
# plt.plot(testPredict[1:])
# plt.show()
