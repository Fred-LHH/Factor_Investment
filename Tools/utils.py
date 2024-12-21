import pandas as pd
import numpy as np
from scipy import stats

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
    if slice:
        rank = factor.groupby('date').rank()
        continuous = p / 2 + (1 - p) * (rank - 1) / (rank.groupby('date').max() - 1)
        def func(series):
            return series.map(lambda x: stats.norm.ppf(x))
        result = 









