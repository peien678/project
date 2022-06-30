# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:15:07 2022

@author: USER
"""

import h5py
import numpy as np
import pandas as pd
import time
import glob
import os
# root_path = 'D:/data/'
# path_list = glob.glob('{root_path}/*/*.txt'.format(**locals()))
# for pth in path_list:
#     os.rename(pth,pth.replace('.txt',''))

# 安装h5py库,导入
di = '2022-04-22'
pdi = di.replace('-','')
Uid = 'eth-usdt'
book_path = 'D:/data/{}-output/{}_binanceUsdtSwap_{}_depth.h5'.format(Uid,pdi,Uid)
trade_path = 'D:/data/{}-output/{}_binanceUsdtSwap_{}_tick.h5'.format(Uid,pdi,Uid)
kline_path = 'D:/data/{}-output/{}_binanceUsdtSwap_{}_kline1m.h5'.format(Uid,pdi,Uid)

f = h5py.File(book_path, 'r')
print(f.keys())
# 读取文件,一定记得加上路径
keys = ['data', 'head', 'timestamp']
book_cols =  np.array(h5py.File(book_path,'r')['head'])
trade_cols =  np.array(h5py.File(trade_path,'r')['head'])
kline_cols =  np.array(h5py.File(kline_path,'r')['head'])
timestamp = np.array(h5py.File(book_path,'r')['timestamp'])
data = np.array(h5py.File(book_path,'r')['data'])
date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp[0]/1e3))
date_time_list = []
for i in range(1000):
    date_time_list.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp[i]/1e3)))