import pandas as pd
import numpy as np
import statsmodels.api as sm
from np_tools import rolling_apply
from joblib import Parallel, delayed
import copy, math

def parallel(df, 
             func, 
             n_core=8):
    len_df = len(df)
    sp = list(range(len_df)[::int(len_df/n_core+0.5)])[:-1] # 最后一个节点改为末尾
    sp.append(len_df)
    slc_gen = (slice(*idx) for idx in zip(sp[:-1],sp[1:]))
    results = Parallel(n_jobs=n_core)(delayed(func)(df[slc]) for slc in slc_gen)
    return pd.concat(results)


