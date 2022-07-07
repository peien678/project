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
from numba import jit
# root_path = 'D:/data/'
# path_list = glob.glob('{root_path}/*/*.txt'.format(**locals()))
# for pth in path_list:
#     os.rename(pth,pth.replace('.txt',''))

# 安装h5py库,导入
# di = '2022-04-21'
# pdi = di.replace('-','')
# Uid = 'eth'
# book_path = 'D:/data/{}-usdt-output/{}_binanceUsdtSwap_{}-usdt_depth.h5'.format(Uid,pdi,Uid)
# trade_path = 'D:/data/{}-usdt-output/{}_binanceUsdtSwap_{}-usdt_tick.h5'.format(Uid,pdi,Uid)
# kline_path = 'D:/data/{}-usdt-output/{}_binanceUsdtSwap_{}-usdt_kline1m.h5'.format(Uid,pdi,Uid)

# f = h5py.File(book_path, 'r')
# print(f.keys())
# # 读取文件,一定记得加上路径
# keys = ['data', 'head', 'timestamp']
# book_cols =  np.array(h5py.File(book_path,'r')['head'])
# trade_cols =  np.array(h5py.File(trade_path,'r')['head'])
# kline_cols =  np.array(h5py.File(kline_path,'r')['head'])
# timestamp = np.array(h5py.File(kline_path,'r')['timestamp'])
# data = np.array(h5py.File(kline_path,'r')['data'])
# date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp[0]/1e3))
# date_time_list = []
# for i in range(1000):
#     date_time_list.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp[i]/1e3)))
# td = TradingDates()
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
td = TradingDates()
class base_feature_lib():
    def __init__(self, di):
        self.root_data_dir = 'D:/data/'
        self.output_dir = 'D:/data/basefeature/'
        self.univ = ['ada','avax','bnb','btc','eth','sol','xrp']
        self.run_mode = 'offline' #parallel
        self.run_list = [1]
        self.data_window = np.array([2,0,1]) #['kline','book','trade']
        self.date = di
        self.delay_ms = 300
        self.kline_front_ms = 5
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
        self.current_min = int((self.timestamp - self.timestamp0)/60000)
        self.current_tod = time.strftime('%H%M%S', time.localtime(self.timestamp/1e3))
        self.calc_mid_rlt('update')
        
        for bfi in self.run_list:
            func_name = 'self.base_feature_%03d'%bfi
            bf_name = eval(func_name)(retn_cf=1)['name']
            dfi = eval(func_name)(retn_cf=0)
            bf_output_dir = '{}{}/{}/'.format(self.output_dir,bf_name,self.date.replace('-','/'))
            dfi.to_csv(bf_output_dir+self.current_tod+'.csv')
        self.timestamp = self.timestamp + 60000
        print(self.timestamp)
        
    def load_di_data_offline(self):
        
        hist = {}
        hist['kline'] = {}
        hist['kline_time'] = {}
        hist['book'] = {}
        hist['book_time'] = {}
        hist['trade'] = {}
        hist['trade_time'] = {}
        pdi = self.date.replace('-', '')
        for Uid in self.univ:
            if self.data_window[0]>0:
                kline_path = '{}{}-usdt-output/{}_binanceUsdtSwap_{}-usdt_kline1m.h5'.format(self.root_data_dir,Uid,pdi,Uid)
                hist['kline'][Uid]      = np.array(h5py.File(kline_path,'r')['data'])
                hist['kline_time'][Uid] = np.array(h5py.File(kline_path,'r')['timestamp'])
            
            if self.data_window[1]>0:
                book_path = '{}{}-usdt-output/{}_binanceUsdtSwap_{}-usdt_depth.h5'.format(self.root_data_dir,Uid,pdi,Uid)
                hist['book'][Uid]      = np.array(h5py.File(book_path,'r')['data'])
                hist['book_time'][Uid] = np.array(h5py.File(book_path,'r')['timestamp'])
            
            if self.data_window[2]>0:
                trade_path = '{}{}-usdt-output/{}_binanceUsdtSwap_{}-usdt_tick.h5'.format(self.root_data_dir,Uid,pdi,Uid)
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
                    kline_path = '{}{}-usdt-output/{}_binanceUsdtSwap_{}-usdt_kline1m.h5'.format(self.root_data_dir,Uid,pdj,Uid)
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
                    book_path = '{}{}-usdt-output/{}_binanceUsdtSwap_{}-usdt_depth.h5'.format(self.root_data_dir,Uid,pdj,Uid)
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
                    trade_path = '{}{}-usdt-output/{}_binanceUsdtSwap_{}-usdt_tick.h5'.format(self.root_data_dir,Uid,pdj,Uid)
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
            hist = self.hist
            m1kline, m1kline_time, m1book, m1book_time, m1trade, m1trade_time = self.get_on_bar_data()
            for Uid in self.univ:
                if self.data_window[0]>0:
                    if Uid in hist['kline'].keys():
                        buf_timestamp = self.timestamp - (self.data_window[0]-1) * 24 * 60 * 60000
                        for i in range(len(hist['kline_time'][Uid])):
                            if hist['kline_time'][Uid][i] > buf_timestamp:
                                break
                        hist['kline'][Uid] = np.vstack([hist['kline'][Uid][i:],m1kline[Uid]])
                        hist['kline_time'][Uid] = np.hstack([hist['kline_time'][Uid][i:],m1kline_time[Uid]])
                    else:
                        hist['kline'][Uid] = m1kline[Uid]
                        hist['kline_time'][Uid] = m1kline_time[Uid]
                if self.data_window[1]>0:
                    if Uid in hist['book'].keys():
                        buf_timestamp = self.timestamp - (self.data_window[1]-1) * 24 * 60 * 60000
                        for i in range(len(hist['book_time'][Uid])):
                            if hist['book_time'][Uid][i] > buf_timestamp:
                                break
                        hist['book'][Uid] = np.vstack([hist['book'][Uid][i:],m1book[Uid]])
                        hist['book_time'][Uid] = np.hstack([hist['book_time'][Uid][i:],m1book_time[Uid]])
                    else:
                        hist['book'][Uid] = m1book[Uid]
                        hist['book_time'][Uid] = m1book_time[Uid]
                if self.data_window[2]>0:
                    if Uid in hist['trade'].keys():
                        buf_timestamp = self.timestamp - (self.data_window[2]-1) * 24 * 60 * 60000
                        for i in range(len(hist['trade_time'][Uid])):
                            if hist['trade_time'][Uid][i] > buf_timestamp:
                                break
                        hist['trade'][Uid] = np.vstack([hist['trade'][Uid][i:],m1trade[Uid]])
                        hist['trade_time'][Uid] = np.hstack([hist['trade_time'][Uid][i:],m1trade_time[Uid]])
                    else:
                        hist['trade'][Uid] = m1trade[Uid]
                        hist['trade_time'][Uid] = m1trade_time[Uid]
            self.hist = hist
        pass
    
    def get_on_bar_data(self):
        self.next_timestamp = self.timestamp+60000
        
        if self.run_mode == 'offline':
            if self.timestamp == int(time.mktime(time.strptime(self.date,'%Y-%m-%d')))*1000:
                min_sep = np.zeros([len(self.univ), 60*24, 3]).astype(int)
                for k in range(len(self.univ)):
                    Uid = self.univ[k]
                    if self.data_window[0]>0:
                        tsptr = self.timestamp - self.kline_front_ms
                        minptr = 0
                        for i in range(len(self.data_di['kline_time'])):
                            if self.data_di['kline_time'][Uid][i] > tsptr:
                                min_sep[k,minptr,0] = i
                                minptr = minptr + 1
                                tsptr  = tsptr  + 60000
                    
                    if self.data_window[1]>0:
                        tsptr = self.timestamp
                        minptr = 0
                        for i in range(len(self.data_di['book_time'])):
                            if self.data_di['book_time'][Uid][i] > tsptr:
                                min_sep[k,minptr,1] = i
                                minptr = minptr + 1
                                tsptr  = tsptr  + 60000
                        
                    if self.data_window[2]>0:
                        tsptr = self.timestamp
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
            min_sep = self.min_sep
            for k in range(len(self.univ)):
                Uid = self.univ[k]
                if self.current_min != 1439:
                    if self.data_window[0]>0:
                        kline_b = min_sep[k,self.current_min,0]
                        kline_e = min_sep[k,self.current_min+1,0]
                        m1kline[Uid] = self.data_di['kline'][Uid][kline_b:kline_e]
                        m1kline_time[Uid] = self.data_di['kline_time'][Uid][kline_b:kline_e]
                    
                    if self.data_window[1]>0:
                        book_b = min_sep[k,self.current_min,1]
                        book_e = min_sep[k,self.current_min+1,1]
                        m1book[Uid] = self.data_di['book'][Uid][book_b:book_e]
                        m1book_time[Uid] = self.data_di['book_time'][Uid][book_b:book_e]
                    
                    if self.data_window[2]>0:
                        trade_b = min_sep[k,self.current_min,2]
                        trade_e = min_sep[k,self.current_min+1,2]
                        m1trade[Uid] = self.data_di['trade'][Uid][trade_b:trade_e]
                        m1trade_time[Uid] = self.data_di['trade_time'][Uid][trade_b:trade_e]
                else:
                    if self.data_window[0]>0:
                        kline_b = min_sep[k,self.current_min,0]
                        m1kline[Uid] = self.data_di['kline'][Uid][kline_b:]
                        m1kline_time[Uid] = self.data_di['kline_time'][Uid][kline_b:]
                    
                    if self.data_window[1]>0:
                        book_b = min_sep[k,self.current_min,1]
                        m1book[Uid] = self.data_di['book'][Uid][book_b:]
                        m1book_time[Uid] = self.data_di['book_time'][Uid][book_b:]
                    
                    if self.data_window[2]>0:
                        trade_b = min_sep[k,self.current_min,2]
                        m1trade[Uid] = self.data_di['trade'][Uid][trade_b:]
                        m1trade_time[Uid] = self.data_di['trade_time'][Uid][trade_b:]
        else:
            pass
        return m1kline, m1kline_time, m1book, m1book_time, m1trade, m1trade_time
    
    def base_feature_001(self, retn_cf = 0):
        if retn_cf == 1:
            config = {}
            config['name'] = 'kline_derivative'
            config['data_window'] = np.array([2,0,0])
            return config
        else:
            cols = ['5M_Open','5M_Close','5M_High','5M_Low','5M_Vwap','5M_Twap',
                    '5M_Dolvol','5M_Bid_Dolvol','5M_Ask_Dolvol',
                    '5M_Raw_Ret']
            univ_tmp = []
            for Uid in self.univ:
                arr =  self.hist['kline'][Uid]
                #['high', 'open', 'low', 'close', 'volume', 'amount', 'bidVolume', 'bidAmount']
                tmp = [
                       arr[-5,1],
                       arr[-1,3],
                       np.max(arr[-5:,0]),
                       np.min(arr[-5:,0]),
                       np.sum(arr[-5:,5])/np.sum(arr[-5:,4]),
                       np.mean(arr[-5:,:4]),
                       np.sum(arr[-5:,5]),
                       np.sum(arr[-5:,7]),
                       np.sum(arr[-5:,5])-np.sum(arr[-5:,7]),
                       arr[-1,3]/arr[-5,1]-1,
                       ]
                univ_tmp.append(tmp)
            output = pd.DataFrame(np.array(univ_tmp).astype(np.float32),index=self.univ,columns=cols)
            output.index.name = 'Uid'
            output['DateTime'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp / 1e3))
            output = output.reindex(columns=['DateTime']+cols)
            return output
    def base_feature_002(self, retn_cf = 0):
        

@jit(nopython=True)
def bf_001():
    return
if __name__ == '__main__':
    ype_bf = base_feature_lib('2022-04-23')
    ype_bf.on_initialize()
    ype_bf.on_notify()
    ype_bf.on_notify()
    ype_bf.on_notify()
