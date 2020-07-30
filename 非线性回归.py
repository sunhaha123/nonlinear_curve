# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:33:20 2020

@author: sunjh
"""

# coding=utf-8

import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from math import e

x_data, y_data = [], []
# 随机生成30个点
for x in range(-20, 20):
    #y= 1/(1+e^(-x))
    y = 1/(1+e**(-x))
    x_data.append(x)
    y_data.append(y)



#x.append()
x_2 = [0,1,0.49,0.51,0.11,0.89]
y_2 = [1,0,0.511,0.111,0.94,0.009]
# 转换为[[]]
# =============================================================================
x_n= np.array(x_data).reshape(1,-1).T
y_n = np.array(y_data).reshape(1,-1).T

#for i  in range(len(y_data)):
#      y_new.append([y_data[i]])
#      x_new.append([x_data[i]])


# 特征构造
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x_n)
# 创建线性模型
linear_reg = LinearRegression()
linear_reg.fit(x_poly, y_data)

plt.plot(x_data, y_data, 'b.')
# 用特征构造数据进行预测
plt.plot(x_data, linear_reg.predict(poly_reg.fit_transform(x_n)), 'r')
plt.show()
