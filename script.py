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
from trading_date import TradingDates
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
td = TradingDates()
class base_feature_lib():
    def __init__(self, di):
        self.root_data_dir = 'D:/data/'
        self.output_dir = 'D:/data/basefeature/'
        self.univ = ['ada','avax','bnb','btc','eth','sol','xrp']
        self.run_mode = 'stream' #parallel
        self.run_list = [1]
        self.data_window = np.array([1,0,0]) #['kline','book','trade']
        self.date = di
    def on_initialize(self):
        for bfi in self.run_list:
            func_name = 'self.base_feature_%03d'%bfi
            self.data_window = np.maximum(self.data_window,eval(func_name)(retn_cf=1)['data_window'])
    def calc_mid_rlt(self, return_type='init'):
        if return_type == 'init':
            hist = {}
            hist['kline'] = {}
            hist['book'] = {}
            hist['trade'] = {}
            for j in range(self.data_window[0]):
                dj = td.prev_tradingday(self.date, j+1)
                pdj = dj.replace('-', '')
                for Uid in self.univ:
                    kline_path = '{}{}-output/{}_binanceUsdtSwap_{}_kline1m.h5'.format(self.root_data_dir,Uid,pdj,Uid)
                    hist_tmp = np.array(h5py.File(kline_path,'r')['data'])
                    if Uid in hist['kline'].keys():
                        hist['kline'][Uid] = np.vstack([hist_tmp,hist['kline'][Uid]])
                    else:
                        hist['kline'][Uid] = hist_tmp
            for j in range(self.data_window[1]):
                dj = td.prev_tradingday(self.date, j+1)
                pdj = dj.replace('-', '')
                for Uid in self.univ:
                    book_path = '{}{}-output/{}_binanceUsdtSwap_{}_depth.h5'.format(self.root_data_dir,Uid,pdj,Uid)
                    hist_tmp = np.array(h5py.File(book_path,'r')['data'])
                    hist_tmp = np.hstack([])
                    if Uid in hist['book'].keys():
                        hist['book'][Uid] = np.vstack([hist_tmp,hist['book'][Uid]])
                    else:
                        hist['book'][Uid] = hist_tmp
        
        pass
    def on_notify(self):
        pass
    
    def base_feature_001(self, retn_cf = 0):
        if retn_cf ==1:
            config = {}
            config['name'] = 'kline_derivative'
            config['data_window'] = np.array([1,0,0])
        else:
            return
        pass