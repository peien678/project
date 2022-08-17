import warnings

import pandas as pd
import numpy as np
import os
import time
import warnings
import boto3
import glob
from config import alpha_config
from trading_date import TradingDates
import h5py

warnings.filterwarnings("ignore")


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def Download_data(date: str, symbol: str):
    root_data_dir = 'D:/data/hftdata/'
    rawdata_path = '{}raw_data/{}/'.format(root_data_dir, date.replace('-', '/'))
    s3 = boto3.resource(
        's3',
        aws_access_key_id='AKIA22GKTJLRFWNPELKH',
        aws_secret_access_key='g3byZwG3SSbocUJrkPaMqpi5+3UEEQvGu5wufask',
        region_name='ap-northeast-1'
    )
    for bucket in s3.buckets.all():
        bucket_name = bucket.name
        break
    bucket = s3.Bucket(bucket_name)
    filters = date + '/' + symbol
    for obj in bucket.objects.filter(Prefix=filters):
        print(obj.key)
        if not os.path.exists(rawdata_path):
            os.makedirs(rawdata_path)
        bucket.download_file(obj.key, rawdata_path + obj.key.split('/')[-1])
    print(f'{symbol} data at {date} download complete')


def compute_base(x_data):
    x_data = x_data.astype(np.float64)
    res = {}
    res['N'] = x_data.notnull().sum(axis=0)
    res['SUM_X'] = x_data.sum(axis=0)
    res['SUM_X2'] = (x_data ** 2).sum(axis=0)
    res['SUM_X3'] = (x_data ** 3).sum(axis=0)
    res['SUM_X4'] = (x_data ** 4).sum(axis=0)
    res['MAX'] = x_data.max(axis=0)
    res['MIN'] = x_data.min(axis=0)
    res['ALL_N'] = x_data.shape[0]
    return pd.DataFrame(res)


def compute_xy_base(xy_data):
    xy_data = xy_data.dropna().astype(np.float64)
    res = {}
    res['N'] = xy_data.shape[0]
    res['SUM_X'] = xy_data.iloc[:, 0].sum(axis=0)
    res['SUM_Y'] = xy_data.iloc[:, 1].sum(axis=0)
    res['SUM_X2'] = (xy_data.iloc[:, 0] ** 2).sum(axis=0)
    res['SUM_Y2'] = (xy_data.iloc[:, 1] ** 2).sum(axis=0)
    res['SUM_XY'] = (xy_data.iloc[:, 0] * xy_data.iloc[:, 1]).sum(axis=0)
    res['QTL20'] = np.nanquantile(xy_data.iloc[:, 0], 0.2)
    res['QTL40'] = np.nanquantile(xy_data.iloc[:, 0], 0.4)
    res['QTL60'] = np.nanquantile(xy_data.iloc[:, 0], 0.6)
    res['QTL80'] = np.nanquantile(xy_data.iloc[:, 0], 0.8)
    pos_ftr = xy_data.iloc[:, 1] > xy_data.iloc[:, 1].median()

    res['IC_POS'] = xy_data[pos_ftr].corr().iloc[0, 1]
    res['IC_NEG'] = xy_data[~pos_ftr].corr().iloc[0, 1]
    fct_rnk = xy_data.iloc[:, 0].rank(pct=True)
    for i in range(10):
        res['Q%dRet' % i] = (xy_data.iloc[:, 1] * (fct_rnk >= i / 10) * (fct_rnk < (i + 1) / 10)).mean(axis=0)
    return pd.Series(res)


def read_y(date, y_names):
    cf = alpha_config()
    pdi = date.replace('-', '')
    y_path = cf.y_path + '%s.h5' % pdi
    y_data = pd.read_hdf(y_path)
    return y_data[['DateTime', 'Uid'] + y_names]


def make_y(date):
    cf = alpha_config()
    di = date
    td = TradingDates()
    pdi = date.replace('-', '')
    dj = td.next_tradingday(di)
    pdj = dj.replace('-', '')
    root_data_dir = cf.root_data_dir
    output_dir = cf.y_path
    y_list = []
    for Uid in cf.ticker_list:
        kline_path_di = '{}bundle_data/{}_binanceUsdtSwap_{}-usdt_kline1m.h5.txt'.format(root_data_dir, pdi, Uid[:-4])
        kline_path_dj = '{}bundle_data/{}_binanceUsdtSwap_{}-usdt_kline1m.h5.txt'.format(root_data_dir, pdj, Uid[:-4])

        kline_di = np.array(h5py.File(kline_path_di, 'r')['data'])
        kline_ts = np.array(h5py.File(kline_path_di, 'r')['timestamp'])
        kline_datetime = []
        for i in range(len(kline_ts)):
            kline_datetime.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime((kline_ts[i] + 1) / 1e3)))
        kline_dj = np.array(h5py.File(kline_path_dj, 'r')['data'])
        kline = np.vstack([kline_di, kline_dj])
        y_names = ['y_180', 'y_180_900', 'y_900_1d']
        y = np.vstack([(kline[3:1440 + 3, 3] - kline[:1440, 3]) / kline[:1440, 3],
                       (kline[15:1440 + 15, 3] - kline[3:1440 + 3, 3]) / kline[3:1440 + 3, 3],
                       (kline[1440:1440 + 1440, 3] - kline[15:1440 + 15, 3]) / kline[15:1440 + 15, 3], ]).T
        y_df = pd.DataFrame(y, index=kline_datetime, columns=y_names)
        y_df.index.name = 'DateTime'
        y_df = y_df.reset_index()
        y_df['Uid'] = Uid
        y_df = y_df.reindex(columns=['DateTime', 'Uid'] + y_names)
        y_list.append(y_df)
    mkdir(output_dir)
    y_df_total = pd.concat(y_list, axis=0, ignore_index=True)
    y_df_total.to_hdf(output_dir + '%s.h5' % pdi, 'data')


def get_factor_daily_stats(date, factor_name, y_names, x_names=None):
    cf = alpha_config()
    pdi = date.replace('-', '/')
    alpha_root_path = cf.alpha_root_path
    factor_save_path = '{}alpha_data/{}/'.format(alpha_root_path, pdi)
    y_path = cf.y_path + '%s.h5' % pdi
    mkdir(factor_save_path)
    if x_names is not None:
        x_names = ['DateTime', 'Uid'] + x_names
    data = pd.read_hdf(factor_save_path + factor_name + '.h5')[x_names]
    y_data = pd.read_hdf(y_path)
