import datetime
import time
import sys

if sys.version_info >= (3, 0):
    unicode = str


def datetime_to_date(dt):
    return datetime.date(dt.year, dt.month, dt.day)


def str_to_date(yyyy_mm_dd):
    return datetime_to_date(datetime.datetime.strptime(yyyy_mm_dd, '%Y-%m-%d'))


def date_to_str(date):
    return date.strftime('%Y-%m-%d')


def int_to_date(d):
    return datetime_to_date(datetime.datetime.strptime(str(d), '%Y%m%d'))


def date_to_int(date):
    if type(date) in (str, unicode):
        return int(date.replace('-', ''))
    return int(date.strftime('%Y%m%d'))


class TradingDates():
    def __init__(self, *vendors):
        if len(vendors) == 0:
            vendors = ['crypto']

    def prev_tradingday(self, date, count=1):

        return date_to_str(str_to_date(date) - datetime.timedelta(days=count))

    def next_tradingday(self, date, count=1):
        return date_to_str(str_to_date(date) + datetime.timedelta(days=count))

    def get_tradingdates(self, begin, end):
        d_list = [begin]
        ptr = begin
        if begin > end:
            raise Exception('end date must be after begin date')
        while ptr != end:
            ptr = self.next_tradingday(ptr)
            d_list.append(ptr)
        return d_list
