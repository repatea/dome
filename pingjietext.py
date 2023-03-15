import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import  pandas as pd
import  os
from keras.models import Sequential, load_model

from sklearn import preprocessing

a = np.array([1, 1, 1])
b = np.array([2, 2, 2])

print('a', a)
print('b', b)
# 将a与b合并
c = np.vstack((a, b))
print('合并结果：\n', c)
print('c的形状：\n', c.shape)


