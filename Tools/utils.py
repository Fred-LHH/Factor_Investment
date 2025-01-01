import pandas as pd
import numpy as np
from scipy import stats
from pd_tools import *

def Rank(factor, 
         norm=False):
    '''
    对因子进行排名
    Params:
    -------
    factor: pd.Series
        Multiindex: date, stock
    
    Returns:
    --------
    每日的因子值从小到大排序, 并均匀映射到(0,1)
    '''
    rank = factor.groupby('date').rank()
    if norm:
        return 2 * 3 **0.5 * (rank / (rank.groupby('date').max() + 1) - 0.5)
    return rank / (rank.groupby('date').max() + 1)

def Norm(factor):
    return (factor - factor.groupby('date').mean()) / factor.groupby('date').std()

def scale(factor,
          a=1):
    return a * factor / factor.groupby('date').apply(lambda x: abs(x).sum())

def Gauss(factor,
          p=0.003,
          slice=False):
    '''
    通过正态分布累计概率函数的逆函数将[p, 1-p]的均匀分布
    转换为正态分布后默认产生3sigma内的样本, 99.7% p=0.003
    '''
    if not slice:
        rank = factor.groupby('date').rank()
        continuous = p / 2 + (1 - p) * (rank - 1) / (rank.groupby('date').max() - 1)
        def func(series):
            return series.map(lambda x: stats.norm.ppf(x))
        result = parallel(continuous, func)
        #如果所有值相同则替换为0
        if_same = result.groupby('date').apply(lambda x: ~(x.duplicated().sum()))
        result.loc[if_same[if_same==1].index] = 0
        return result
    else:
        rank = factor.rank()
        continuous = p / 2 + (1 - p) * (rank - 1) / (rank.max() - 1)
        return continuous.map(lambda x: stats.norm.ppf(x))

def resample_fill(factor, 
                  freq='month'):
    '''
    每月、每周内的因子值替换为月初、周初的因子值
    '''
    factor.name = 0
    df = pd.DataFrame(factor)
    df['th'] = df.index.map(lambda x: getattr(x[0], freq))
    df['yesterday_th'] = df['th'].groupby('code').shift()
    df = df.fillna(getattr(df.index[0][0], freq))
    df['after'] = df.apply(lambda x: x[0] if x.th != x.yesterday_th else np.nan, axis=1)
    df = df.groupby('code').fillna(method='ffill')
    df = df.groupby('code').fillna(method='bfill')
    return df['after']

def resample_select(market,
                    freq='month'):
    '''
    直接使用原数据的周初、月初值
    '''
    if freq=='month':
        # 前一天月份和今天不同，后一天月份和今天相同
        market['th'] = market.index.get_level_values(0).map(lambda x: x.month)
    elif freq=='week':
        market['th'] = market.index.get_level_values(0).map(lambda x: x.week)
    return market[(~market['th'].groupby('code').shift().isna())&\
                  (market['th'].groupby('code').shift()!=market['th'])&\
                  (market['th'].groupby('code').shift(-1)==market['th'])].drop(columns='th')







