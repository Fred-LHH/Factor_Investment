from functools import partial
from typing import Callable, Any, Union, Generator, Tuple, List
import numpy as np
from joblib import Parallel, delayed

Number = Union[int, float]


def expstep_range(
        start: Number,
        end: Number,
        min_step: Number=1,
        step_mult: Number=1,
        round_func: Callable=None
        ) -> np.ndarray:
    '''
    Params:
    -------
    start: int or float
        起始值
    end: int or float
        结束值
    min_step: int or float
        最小步长
    step_mult: int or float
        步长倍数
    round_func: Callable, optional
        eg. np.round, np.ceil, np.floor

    Returns:
    --------
    np.ndarray
    '''
    
    if step_mult <= 0:
        raise ValueError('multi_step should be bigger than 0')
    
    if min_step <= 0:
        raise ValueError('min_step should be bigger than 0')
    
    last = start
    values = []
    step = min_step

    sign = 1 if start < end else -1

    while start < end and last < end or start > end and last > end:
        values.append(last)
        last += max(step, min_step) * sign
        step = abs(step * step_mult)

    values = np.array(values)
    if not round_func:
        return values
    
    values = np.array(round_func(values))
    _, idx = np.unique(values, return_index=True)
    
    return values[np.sort(idx)]
    

def apply_map(
    func: Callable[[Any], Any],
    data: Union[np.ndarray, List]
    ) -> np.ndarray:
    '''
    Params:
    -------
    func: Callable
        接受输入一个参数的函数并返回一个值
    data: np.ndarray or List
        输入数据
    
    Returns:
    --------
    np.ndarray
    '''
    # convert list to np.ndarray first
    array = np.array(data)
    array_view = array.flat
    array_view[:] = [func(x) for x in array_view]
    return array

def nans(
        shape: Union[int, Tuple[int, ...]],
        dtype=float
        ) -> np.ndarray:
    '''
    返回一个新的array, 对于给定的shape填充缺失值
    '''
    if np.issubdtype(dtype, np.integer):
        dtype = float
    arr = np.empty(shape, dtype=dtype)
    if np.issubtype(arr.dtype, np.datetime64):
        arr.fill(np.datetime64('NaT'))
    else:
        arr.fill(np.nan)
    return arr

def drop_na(array: np.ndarray) -> np.ndarray:
    '''
    Params:
    -------
    array: np.ndarray
        Input array
    
    Returns:
    --------
    np.ndarray
        return a given array flattened and with nans dropped
    '''
    return array[~np.isnan(array)]

def fill_na(array: np.ndarray, 
            value: Any
            ) -> np.ndarray:
    '''
    Params:
    -------
    array: np.ndarray
        Input array
    value: Any
        Value to fill nans
    
    Returns:
    --------
    np.ndarray
        return a given array with nans filled
    '''
    ar = array.copy()
    ar[np.isnan(ar)] = value
    return ar









