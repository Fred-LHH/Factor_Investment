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

def fill_not_finite(array: np.ndarray, 
                    value: Any = 0) -> np.ndarray:
    """
    Params
    ----------
    array : np.ndarray
        Input array.
    value : Any, optional
        Value to replace nans and infs with. Default is 0.

    Returns
    -------
    np.ndarray
        A copy of array with nans and infs replaced with the given value.
    """
    ar = array.copy()
    ar[~np.isfinite(array)] = value
    return ar

def prepend_na(array: np.ndarray, 
               n: int) -> np.ndarray:
    """
    Params
    ----------
    array : np.ndarray
        Input array.
    n : int
        Number of elements to insert.

    Returns
    -------
    np.ndarray
        New array with nans added at the beginning.
    """
    if not len(array):  # if empty, simply create empty array
        return np.hstack((nans(n), array))

    elem = array[0]
    dtype = float
    if hasattr(elem, 'dtype'):
        dtype = elem.dtype

    if hasattr(elem, '__len__') and len(elem) > 1:  # if the array has many dimension
        if isinstance(array, np.ndarray):
            array_shape = array.shape
        else:
            array_shape = np.array(array).shape
        return np.vstack((nans((n, *array_shape[1:]), dtype), array))
    else:
        return np.hstack((nans(n, dtype), array))

def rolling(
    array: np.ndarray,
    window: int,
    skip_na: bool = False,
    as_array: bool = False
) -> Union[Generator[np.ndarray, None, None], np.ndarray]:
    """
    Params
    ----------
    array : np.ndarray
        Input array.
    window : int
        Size of the rolling window.
    skip_na : bool, optional
        If False, the sequence starts with (window-1) windows filled with nans. If True, those are omitted.
        Default is False.
    as_array : bool, optional
        If True, return a 2-D array. Otherwise, return a generator of slices. Default is False.

    Returns
    -------
    np.ndarray or Generator[np.ndarray, None, None]
        Rolling window matrix or generator
    """
    if not any(isinstance(window, t) for t in [int, np.integer]):
        raise TypeError(f'Wrong window type ({type(window)}) int expected')

    window = int(window)

    if array.size < window:
        raise ValueError('array.size should be bigger than window')

    def rows_gen():
        if not skip_na:
            yield from (prepend_na(array[:i + 1], (window - 1) - i) for i in np.arange(window - 1))

        starts = np.arange(array.size - (window - 1))
        yield from (array[start:end] for start, end in zip(starts, starts + window))

    return np.array([row for row in rows_gen()]) if as_array else rows_gen()

def rolling_apply(
    func: Callable,
    window: int,
    *arrays: np.ndarray,
    prepend_nans: bool = True,
    n_jobs: int = 1,
    **kwargs
) -> np.ndarray:
    """
    Params
    ----------
    func : Callable
        The function to apply to each slice or a group of slices.
    window : int
        Window size.
    *arrays : list
        List of input arrays.
    prepend_nans : bool
        Specifies if nans should be prepended to the resulting array
    n_jobs : int, optional
        Parallel tasks count for joblib. If 1, joblib won't be used. Default is 1.
    **kwargs : dict
        Input parameters (passed to func, must be named).

    Returns
    -------
    np.ndarray
    """
    if not any(isinstance(window, t) for t in [int, np.integer]):
        raise TypeError(f'Wrong window type ({type(window)}) int expected')

    window = int(window)

    if max(len(x.shape) for x in arrays) != 1:
        raise ValueError('Wrong array shape. Supported only 1D arrays')

    if len({array.size for array in arrays}) != 1:
        raise ValueError('Arrays must be the same length')

    def _apply_func_to_arrays(idxs):
        return func(*[array[idxs[0]:idxs[-1] + 1] for array in arrays], **kwargs)

    array = arrays[0]
    rolls = rolling(
        array if len(arrays) == n_jobs == 1 else np.arange(len(array)),
        window=window,
        skip_na=True
    )

    if n_jobs == 1:
        if len(arrays) == 1:
            arr = list(map(partial(func, **kwargs), rolls))
        else:
            arr = list(map(_apply_func_to_arrays, rolls))
    else:
        f = delayed(_apply_func_to_arrays)
        arr = Parallel(n_jobs=n_jobs)(f(idxs[[0, -1]]) for idxs in rolls)

    return prepend_na(arr, n=window - 1) if prepend_nans else np.array(arr)

def expanding(
    array: np.ndarray,
    min_periods: int = 1,
    skip_na: bool = True,
    as_array: bool = False
) -> Union[Generator[np.ndarray, None, None], np.ndarray]:
    """
    Params
    ----------
    array : np.ndarray
        Input array.
    min_periods : int, optional
        Minimum size of the window. Default is 1.
    skip_na : bool, optional
        If False, the windows of size less than min_periods are filled with nans. If True, they're dropped.
        Default is True.
    as_array : bool, optional
        If True, return a 2-D array. Otherwise, return a generator of slices. Default is False.

    Returns
    -------
    np.ndarray or Generator[np.ndarray, None, None]
    """
    if not any(isinstance(min_periods, t) for t in [int, np.integer]):
        raise TypeError(f'Wrong min_periods type ({type(min_periods)}) int expected')

    min_periods = int(min_periods)

    if array.size < min_periods:
        raise ValueError('array.size should be bigger than min_periods')

    def rows_gen():
        if not skip_na:
            yield from (nans(i) for i in np.arange(1, min_periods))

        yield from (array[:i] for i in np.arange(min_periods, array.size + 1))

    return np.array([row for row in rows_gen()], dtype=object) if as_array else rows_gen()


def expanding_apply(
    func: Callable,
    min_periods: int,
    *arrays: np.ndarray,
    prepend_nans: bool = True,
    n_jobs: int = 1,
    **kwargs
) -> np.ndarray:
    """
    Params
    ----------
    func : Callable
        The function to apply to each slice or a group of slices.
    min_periods : int
        Minimal size of expanding window.
    *arrays : list
        List of input arrays.
    prepend_nans : bool
        Specifies if nans should be prepended to the resulting array
    n_jobs : int, optional
        Parallel tasks count for joblib. If 1, joblib won't be used. Default is 1.
    **kwargs : dict
        Input parameters (passed to func, must be named).

    Returns
    -------
    np.ndarray
    """
    if not any(isinstance(min_periods, t) for t in [int, np.integer]):
        raise TypeError(f'Wrong min_periods type ({type(min_periods)}) int expected')

    min_periods = int(min_periods)

    if max(len(x.shape) for x in arrays) != 1:
        raise ValueError('Supported only 1-D arrays')

    if len({array.size for array in arrays}) != 1:
        raise ValueError('Arrays must be the same length')

    def _apply_func_to_arrays(idxs):
        return func(*[array[idxs.astype(int)] for array in arrays], **kwargs)

    array = arrays[0]
    rolls = expanding(
        array if len(arrays) == n_jobs == 1 else np.arange(len(array)),
        min_periods=min_periods,
        skip_na=True
    )

    if n_jobs == 1:
        if len(arrays) == 1:
            arr = list(map(partial(func, **kwargs), rolls))
        else:
            arr = list(map(_apply_func_to_arrays, rolls))
    else:
        f = delayed(_apply_func_to_arrays)
        arr = Parallel(n_jobs=n_jobs)(map(f, rolls))

    return prepend_na(arr, n=min_periods - 1) if prepend_nans else np.array(arr)


