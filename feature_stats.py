import warnings

import pandas as pd
import numpy as np
import os
import time
import warnings
import boto3
import glob


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
    pdi = date.replace('-', '/')
    ROOT_SAVE_PATH = 'D:/data/hftalpha/'
    y_path = '{}y_data/{}/'.format(ROOT_SAVE_PATH, pdi)

    pass
def make_y(date):
    pdi = date.replace('-', '/')
    ROOT_SAVE_PATH = 'D:/data/hftalpha/'
    y_path = '{}y_data/{}/'.format(ROOT_SAVE_PATH, pdi)
    mkdir(y_path)
    tkr_list = ['adausdt', 'avaxusdt', 'bnbusdt', 'btcusdt', 'dogeusdt', 'dotusdt', 'ethusdt', 'maticusdt', 'solusdt',
                'xrpusdt']
    for tkr in tkr_list:
        bdldata_path = '{}raw_data/{}/'.format(ROOT_SAVE_PATH, pdi)



def get_factor_daily_stats(date, factor_name, x_names, y_names):
    pdi = date.replace('-', '/')
    ROOT_SAVE_PATH = 'D:/data/hftalpha/'
    factor_save_path = '{}alpha_data/{}/'.format(ROOT_SAVE_PATH, pdi)
    mkdir(factor_save_path)
