# data=load('loss.txt')
# A=data(:,1)
# B=data(:,2)
# plot(A,'g','linewidth',2)
# hold on
# plot(B,'r','linewidth',2)
# legend('train-Loss','val-Loss')
# encoding: utf-8
"""
Created on 2019.09.07
@Author:    MK
@Contact:   makangemail@126.com
@Blog:      https://blog.csdn.net/weixin_44100850
@Filename:  draw.py
@Description:
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def gaussian(x, *param):
    return param[0] * np.exp(-np.power(x - param[2], 2.) / (2 * np.power(param[4], 2.))) + \
           param[1] * np.exp(-np.power(x - param[3], 2.) / (2 * np.power(param[5], 2.)))


# class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = {'batch':[], 'epoch':[]}
#         self.accuracy = {'batch':[], 'epoch':[]}
#         self.val_loss = {'batch':[], 'epoch':[]}
#         self.val_acc = {'batch':[], 'epoch':[]}

#     def on_batch_end(self, batch, logs={}):
#         self.losses['batch'].append(logs.get('loss'))
#         self.accuracy['batch'].append(logs.get('acc'))
#         self.val_loss['batch'].append(logs.get('val_loss'))
#         self.val_acc['batch'].append(logs.get('val_acc'))

#     def on_epoch_end(self, batch, logs={}):
#         self.losses['epoch'].append(logs.get('loss'))
#         self.accuracy['epoch'].append(logs.get('acc'))
#         self.val_loss['epoch'].append(logs.get('val_loss'))
#         self.val_acc['epoch'].append(logs.get('val_acc'))

#     def loss_plot(self, loss_type):
#         iters = range(len(self.losses[loss_type]))
#         plt.figure()
#         # acc
#         plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
#         # loss
#         plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
#         if loss_type == 'epoch':
#             # val_acc
#             plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
#             # val_loss
#             plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
#         plt.grid(True)
#         plt.xlabel(loss_type)
#         plt.ylabel('acc-loss')
#         plt.legend(loc="upper right")
#         plt.show()


file = open('miou_acc.csv')  # 打开文档
data = file.readlines()  # 读取文档数据

para_1 = []  #######表格第一列代表迭代数
para_2 = []  #######第二列代表train——loss
# para_3 = []  #######第三列代表val_loss
# para_4 = []  #新建列表，用于保存第二列数据
# para_5 = []  #新建列表，用于保存第二列数据
for num in data:
    # split用于将每一行数据用逗号分割成多个对象
    # 取分割后的第0列，转换成float格式后添加到para_1列表中
    para_2.append(float(num.split(',')[0]))
    # 取分割后的第1列，转换成float格式后添加到para_1列表中
#     para_2.append(float(num.split(',')[1]))
#     para_3.append(float(num.split(',')[2]))
#     para_4.append(float(num.split(',')[3]))
#     para_5.append(float(num.split(',')[4]))

t = len(para_2)
for i in range(t):
    para_1.append(i)
print(para_1)

popt, pcov = curve_fit(gaussian, para_1, para_2, p0=[3, 4, 3, 6, 1, 1], maxfev=500000)
# popt1,pcov1 = curve_fit(gaussian,para_1,para_3,p0=[3,4,3,6,1,1],maxfev=500000)
plt.figure()

# acc
# plt.plot(para_1, para_2, 'k', label='train_loss' )#r对应颜色   A对应的标签名字
plt.plot(para_1, gaussian(para_1, *popt), 'g')
# plt.plot(para_1,gaussian(para_1,*popt1),'b',label='b')
# loss
# plt.plot(para_1, para_3, 'b', label='val_loss')
# plt.plot(para_1, para_4, 'g', label='c')
# plt.plot(para_1, para_5, 'k', label='PSPNet50')

# plt.grid(True)
plt.xlabel('Iter')  #####xy轴标签
plt.ylabel('miou')
plt.legend(loc='upper right', fontsize=5)
plt.savefig("miou.png", dpi=2000)
plt.show()
# plt.title('map')
# plt.plot(para_1, para_2)
# plt.show()
# !/usr/bin/python
# -*- coding: UTF-8 -*-
# import PIL.Image as Img
# import os
# import pickle
# import xml.etree.ElementTree as ET
# from xml.etree.ElementTree import ElementTree, Element
# import matplotlib.pyplot as plt
# import time

# data_dir = 'loss.txt'
# f1 = open(data_dir, 'r', encoding='utf-8')
# x = []
# y = []
# y1 = []
# for line in f1:
#     line = line.split()
#     if 'jueyuanzi_01' in line and 'accuracy' in line[4]:
#         #print(line[4].split(':')[1])
#         y.append(float(line[4].split(':')[1]))
#     if 'jueyuanzi_02' in line and 'accuracy' in line[4]:
#         #print(line[4].split(':')[1])
#         y1.append(float(line[4].split(':')[1]))
#     if 'Start' in line and 'Epoch' in line and int(line[5]) % 5 == 0 and line[5] != '5' and int(line[5])>=50:
#         # print(int(line[5]))
#         x.append(int(line[5]))

# plt.plot(x, y, label='jueyuanzi_01')
# plt.plot(x, y1, label='jueyuanzi_02')
# plt.legend(loc='best')
# # plt.xlabel('Epoch')
# plt.show()