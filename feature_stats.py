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
    pos_ftr = xy_data.iloc[:, 0] > xy_data.iloc[:, 0].median()

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
    data = data.merge(y_data, on=['DateTime', 'Uid'], how='left')
    data = data.replace([np.inf, -np.inf], np.nan)
    factor_stats_path = '{}factor_daily_stats/{}/'.format(alpha_root_path, pdi)
    data = data.merge(y_data, on=['DateTime', 'Uid'], how='left')
    data = data.replace([np.inf, -np.inf], np.nan)
    factor_stats_path = '{}factor_daily_stats/{}/'.format(alpha_root_path, factor_name)
    mkdir(factor_stats_path)
    single_factor_stats = compute_base(data[x_names])
    tmp_save_path = '{}/single_stats_{}.pkl'.format(factor_stats_path, date)
    res = {}
    for y_name in y_names:
        for x_name in x_names:
            try:
                tmp_single_data = data[[x_name, y_name]].dropna()
            except:
                print(x_name, y_name, 'error')
                continue
            res[(y_name, x_name)] = compute_xy_base(tmp_single_data).astype(np.float64)
    res = pd.DataFrame(res).T
    cov_xy = (res['SUM_XY'] - res['SUM_X'] * res['SUM_Y'] / res['N']) / res['N']
    var_x = (res['SUM_X2'] - res['SUM_X'] * res['SUM_X'] / res['N']) / res['N']
    var_y = (res['SUM_Y2'] - res['SUM_Y'] * res['SUM_Y'] / res['N']) / res['N']
    beta = cov_xy / var_x
    alpha = res['SUM_Y'] / res['N'] - beta * res['SUM_X'] / res['N']
    ic = cov_xy / np.sqrt(var_x * var_y)
    res['DailyBeta'] = beta
    res['DailyAlpha'] = alpha
    res['DailyIC'] = ic
    tmp_save_path = '{}/xy_stats_{}.pkl'.format(factor_stats_path, date)
    res.to_pickle(tmp_save_path)

    corr = data.corr()
    tmp_save_path = '{}/factor_corr_{}.pkl'.format(factor_stats_path, date)
    corr.to_pickle(tmp_save_path)
    print('daily_stats done in ', date, 'for ', factor_name)


def comb_daily_stats(date_list, daily_save_path):
    xy_stats = []
    single_stats = []
    corr = []
    n = 0
    tmp_path = daily_save_path
    for date in date_list:
        try:
            file_name = 'single_stats_{}.pkl'.format(date)
            tmp_file_path = tmp_path + file_name
            single_factor_stats = pd.read_pickle(tmp_file_path)
            file_name = 'xy_stats_{}.pkl'.format(date)
            tmp_file_path = tmp_path + file_name
            xy_factor_stats = pd.read_pickle(tmp_file_path)
            file_name = 'factor_corr_{}.pkl'.format(date)
            tmp_file_path = tmp_path + file_name
            factor_corr = pd.read_pickle(tmp_file_path)
        except:
            print(date, 'error')
            continue
        xy_factor_stats.index.name = ['Y_NAME', 'X_NAME']
        xy_factor_stats = xy_factor_stats.reset_index()
        xy_factor_stats['DATE'] = date
        xy_stats.append(xy_factor_stats)

        single_factor_stats.index.name = 'X_NAME'
        single_factor_stats = single_factor_stats.reset_index()
        single_factor_stats['DATE'] = date
        single_stats.append(single_factor_stats)

        factor_corr.index.name = 'X_NAME'
        factor_corr = factor_corr.reset_index()
        corr.append(factor_corr)
        n += 1
    xy_stats = pd.concat(xy_stats)
    single_stats = pd.concat(single_stats)
    corr = pd.concat(corr)

    comb_path = tmp_path + 'comb_single_stats.h5'
    try:
        single_stats_old = pd.read_hdf(comb_path)
        single_stats_new = pd.concat([single_stats_old, single_stats]).drop_duplicates(['X_NAME', 'DATE']).sort_values(
            'DATE')
    except:
        if os.path.exists(comb_path):
            os.remove(comb_path)
        single_stats_new = single_stats
    single_stats_new.to_hdf(comb_path)

    comb_path = tmp_path + 'comb_xy_stats.h5'
    try:
        xy_stats_old = pd.read_hdf(comb_path)
        xy_stats_new = pd.concat([xy_stats_old, xy_stats]).drop_duplicates(['X_NAME', 'Y_NAME', 'DATE']).sort_values(
            'DATE')
    except:
        if os.path.exists(comb_path):
            os.remove(comb_path)
        xy_stats_new = xy_stats
    xy_stats_new.to_hdf(comb_path)

    comb_path = tmp_path + 'comb_corr.h5'
    try:
        corr_old = pd.read_hdf(comb_path)
        corr_new = pd.concat([corr_old, corr]).drop_duplicates(['X_NAME', 'DATE']).sort_values(
            'DATE')
    except:
        if os.path.exists(comb_path):
            os.remove(comb_path)
        corr_new = corr
    corr_new.to_hdf(comb_path)

    all_res = {}
    all_res['Max'] = single_stats.groupby('X_NAME')['MAX'].max()
    all_res['Min'] = single_stats.groupby('X_NAME')['MIN'].min()

    sum_cols = ['N', 'SUM_X', 'SUM_X2', 'SUM_X3', 'SUM_X4', 'ALL_N']
    res = single_stats.groupby('X_NAME')[sum_cols].sum().astype(np.float64)
    mean_x = res['SUM_X'] / res['N']
    var_x = res['SUM_X2'] / res['N'] - mean_x ** 2
    std_x = np.sqrt(var_x)
    skew_x = (res['SUM_X3'] / res['N'] - 3 * mean_x * var_x - mean_x ** 3) / np.power(var_x, 3 / 2)
    kurt_x = (res['SUM_X4'] / res['N'] - 4 * res['SUM_X3'] / res['N'] * mean_x + 2 * res['SUM_X2'] / res[
        'N'] * mean_x ** 2 + 4 * var_x * mean_x ** 2 + mean_x ** 4) / np.power(var_x, 2) - 3

    all_res['Mean'] = mean_x
    all_res['Std'] = std_x
    all_res['Skew'] = skew_x
    all_res['Kurt'] = kurt_x
    all_res['Count'] = res['N']
    all_res['ValidCountRatio'] = res['N'] / res['ALL_N']
    all_res = pd.DataFrame(all_res)
    single_res = all_res.copy()

    all_res = {}
    sum_cols = ['N', 'SUM_X', 'SUM_X2', 'SUM_Y', 'SUM_Y2', 'SUM_XY']
    res = xy_stats.groupby(['X_NAME', 'Y_NAME'])[sum_cols].sum().astype(np.float64)
    mean_y = res['SUM_Y'] / res['N']
    cov_xy = res['SUM_XY'] / res['N'] - mean_y * mean_x
    var_y = res['SUM_Y2'] / res['N'] - mean_y ** 2
    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    ic = cov_xy / np.sqrt(var_x * var_y)

    all_res['Beta'] = beta
    all_res['Alpha'] = alpha
    all_res['IC'] = ic
    all_res['R2'] = ic ** 2 * 1e4
    all_res = pd.DataFrame(all_res)

    daily_cols = ['IC_POS', 'IC_NEG', 'QTL20', 'QTL40', 'QTL60', 'QTL80'] + ['Q%dRet' % i for i in range(10)]
    tmp_res = xy_stats.groupby(['X_NAME', 'Y_NAME'])[daily_cols].mean().astype(np.float64)
    all_res = pd.concat([all_res, tmp_res], axis=1)

    daily_cols = ['DailyBeta', 'DailyAlpha', 'DailyIC']
    tmp_res = xy_stats.groupby(['X_NAME', 'Y_NAME'])[daily_cols].mean() / xy_stats.groupby(['X_NAME', 'Y_NAME'])[
        daily_cols].std(ddof=0).astype(np.float64)
    tmp_res.columns = [s + '_sharpe' for s in tmp_res.columns]
    all_res = pd.concat([all_res, tmp_res], axis=1)

    all_res = all_res.join(single_res)
    corr = corr.groupby(['X_NAME']).mean().reindex(index=corr.columns[1:-1]).astype(np.float64)
    daily_stats = xy_stats[['X_NAME', 'Y_NAME', 'DailyAlpha', 'DailyBeta', 'DailyIC', 'DATE']]
    return all_res, corr, daily_stats
