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
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

class base_feature_lib():
    def __init__(self, di):
        self.root_data_dir = 'D:/data/'
        self.output_dir = 'D:/data/basefeature/'
        self.univ = ['ada','avax','bnb','btc','eth','sol','xrp']
        self.run_mode = 'offline' #parallel
        self.run_list = [1]
        self.data_window = np.array([2,0,1]) #['kline','book','trade']
        self.date = di
        self.delay_ms = 500
        self.timestamp = int(time.mktime(time.strptime(di,'%Y-%m-%d')))*1000
        self.timestamp0 = int(time.mktime(time.strptime(di,'%Y-%m-%d')))*1000
    def on_initialize(self):
        for bfi in self.run_list:
            func_name = 'self.base_feature_%03d'%bfi
            self.data_window = np.maximum(self.data_window,eval(func_name)(retn_cf=1)['data_window'])
        self.calc_mid_rlt('init')
        if self.run_mode == 'offline':
            self.load_di_data_offline()
            
        for bfi in self.run_list:
            func_name = 'self.base_feature_%03d'%bfi
            bf_name = eval(func_name)(retn_cf=1)['name']
            bf_output_dir = '{}{}/{}/'.format(self.output_dir,bf_name,self.date.replace('-','/'))
            mkdir(bf_output_dir)
    def on_notify(self):
        self.timestamp = self.timestamp + 60000
        self.current_min = int((self.timestamp - self.timestamp0)/60000)
        self.current_tod = time.strftime('%H:%M:%S', time.localtime(self.timestamp/1e3))
        self.calc_mid_rlt('update')
        
        for bfi in self.run_list:
            func_name = 'self.base_feature_%03d'%bfi
            bf_name = eval(func_name)(retn_cf=1)['name']
            dfi = eval(func_name)(retn_cf=0)
            bf_output_dir = '{}{}/{}/'.format(self.output_dir,bf_name,self.date.replace('-','/'))
            dfi.to_csv(bf_output_dir+self.current_tod+'.csv')
        pass
    def load_di_data_offline(self):
        
        hist = {}
        hist['kline'] = {}
        hist['kline_time'] = {}
        hist['book'] = {}
        hist['book_time'] = {}
        hist['trade'] = {}
        hist['trade_time'] = {}
        pdi = di.replace('-', '')
        for Uid in self.univ:
            if self.data_window[0]>0:
                kline_path = '{}{}-output/{}_binanceUsdtSwap_{}_kline1m.h5'.format(self.root_data_dir,Uid,pdi,Uid)
                hist['kline'][Uid]      = np.array(h5py.File(kline_path,'r')['data'])
                hist['kline_time'][Uid] = np.array(h5py.File(kline_path,'r')['timestamp'])
            
            if self.data_window[1]>0:
                book_path = '{}{}-output/{}_binanceUsdtSwap_{}_depth.h5'.format(self.root_data_dir,Uid,pdi,Uid)
                hist['book'][Uid]      = np.array(h5py.File(book_path,'r')['data'])
                hist['book_time'][Uid] = np.array(h5py.File(book_path,'r')['timestamp'])
            
            if self.data_window[2]>0:
                trade_path = '{}{}-output/{}_binanceUsdtSwap_{}_tick.h5'.format(self.root_data_dir,Uid,pdi,Uid)
                hist['trade'][Uid]      = np.array(h5py.File(trade_path,'r')['data'])
                hist['trade_time'][Uid] = np.array(h5py.File(trade_path,'r')['timestamp'])
        self.data_di = hist
    def calc_mid_rlt(self, return_type='init'):
        if return_type == 'init':
            hist = {}
            hist['kline'] = {}
            hist['kline_time'] = {}
            hist['book'] = {}
            hist['book_time'] = {}
            hist['trade'] = {}
            hist['trade_time'] = {}
            for j in range(max(0, self.data_window[0]-1)):
                dj = td.prev_tradingday(self.date, j+1)
                pdj = dj.replace('-', '')
                for Uid in self.univ:
                    kline_path = '{}{}-output/{}_binanceUsdtSwap_{}_kline1m.h5'.format(self.root_data_dir,Uid,pdj,Uid)
                    hist_tmp = np.array(h5py.File(kline_path,'r')['data'])
                    time_tmp = np.array(h5py.File(kline_path,'r')['timestamp'])
                    if Uid in hist['kline'].keys():
                        hist['kline'][Uid] = np.vstack([hist_tmp,hist['kline'][Uid]])
                        hist['kline_time'][Uid] = np.hstack([time_tmp,hist['kline_time'][Uid]])
                    else:
                        hist['kline'][Uid] = hist_tmp
                        hist['kline_time'][Uid] = time_tmp
            for j in range(max(0, self.data_window[1]-1)):
                dj = td.prev_tradingday(self.date, j+1)
                pdj = dj.replace('-', '')
                for Uid in self.univ:
                    book_path = '{}{}-output/{}_binanceUsdtSwap_{}_depth.h5'.format(self.root_data_dir,Uid,pdj,Uid)
                    hist_tmp = np.array(h5py.File(book_path,'r')['data'])
                    time_tmp = np.array(h5py.File(book_path,'r')['timestamp'])
                    if Uid in hist['book'].keys():
                        hist['book'][Uid] = np.vstack([hist_tmp,hist['book'][Uid]])
                        hist['book_time'][Uid] = np.hstack([time_tmp,hist['book_time'][Uid]])
                    else:
                        hist['book'][Uid] = hist_tmp
                        hist['book_time'][Uid] = time_tmp
            for j in range(max(0, self.data_window[2]-1)):
                dj = td.prev_tradingday(self.date, j+1)
                pdj = dj.replace('-', '')
                for Uid in self.univ:
                    trade_path = '{}{}-output/{}_binanceUsdtSwap_{}_tick.h5'.format(self.root_data_dir,Uid,pdj,Uid)
                    hist_tmp = np.array(h5py.File(trade_path,'r')['data'])
                    time_tmp = np.array(h5py.File(trade_path,'r')['timestamp'])
                    if Uid in hist['trade'].keys():
                        hist['trade'][Uid] = np.vstack([hist_tmp,hist['trade'][Uid]])
                        hist['trade_time'][Uid] = np.hstack([time_tmp,hist['trade_time'][Uid]])
                    else:
                        hist['trade'][Uid] = hist_tmp
                        hist['trade_time'][Uid] = time_tmp
            self.hist = hist
        else:
            m1kline, m1kline_time, m1book, m1book_time, m1trade, m1trade_time = self.get_on_bar_data()
        pass
    def get_on_bar_data(self):
        self.next_timestamp = self.timestamp+60000
        
        if self.run_mode == 'offline':
            if self.timestamp == int(time.mktime(time.strptime(self.date,'%Y-%m-%d')))*1000:
                min_sep = np.zeros([len(self.univ), 60*24, 3]).astype(int)
                for k in range(len(self.univ)):
                    Uid = self.univ[k]
                    tsptr = self.timestamp - self.delay_ms
                    minptr = 0
                    for i in range(len(self.data_di['kline_time'])):
                        if self.data_di['kline_time'][Uid][i] > tsptr:
                            min_sep[k,minptr,0] = i
                            minptr = minptr + 1
                            tsptr  = tsptr  + 60000
                    
                    tsptr = self.timestamp - self.delay_ms
                    minptr = 0
                    for i in range(len(self.data_di['book_time'])):
                        if self.data_di['book_time'][Uid][i] > tsptr:
                            min_sep[k,minptr,1] = i
                            minptr = minptr + 1
                            tsptr  = tsptr  + 60000
                        
                    tsptr = self.timestamp - self.delay_ms
                    minptr = 0
                    for i in range(len(self.data_di['trade_time'])):
                        if self.data_di['trade_time'][Uid][i] > tsptr:
                            min_sep[k,minptr,2] = i
                            minptr = minptr + 1
                            tsptr  = tsptr  + 60000
                self.min_sep = min_sep
            m1kline      = {}
            m1kline_time = {}
            m1book       = {}
            m1book_time  = {}
            m1trade      = {}
            m1trade_time = {}
            currnet_min = int(self.timestamp/self.timestamp0)
            for k in range(len(self.univ)):
                Uid = self.univ[k]
                m1kline = min_sep
    
    def base_feature_001(self, retn_cf = 0):
        if retn_cf ==1:
            config = {}
            config['name'] = 'kline_derivative'
            config['data_window'] = np.array([1,0,0])
        else:
            return
        pass
if __name__ == '__main__':
    ype_bf = base_feature_lib('2022-04-22')
    