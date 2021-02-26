from typing import Union

import numpy as np
import tensorflow as tf


def create_train_test_splits(*args: Union[np.ndarray, tf.Tensor, list], ratio: float = 0.3, ):
    assert len(args) > 0, "please supply at least one data container"
    len_arr = get_number_obs(args[0])
    for arg in args:
        assert len_arr == get_number_obs(arg), "All data containers must have the same shape or length"
    test_size = int(ratio * len_arr)
    indices = np.random.choice(np.arange(len_arr), size=test_size, replace=False)
    mask = np.ones(len_arr, dtype=bool)
    mask[indices] = False

    splits = []
    for arg in args:
        if isinstance(arg, list):
            arg = np.array(arg)
            split = arg[mask].to_list(), arg[~mask].to_list()
        else:
            split = arg[mask], arg[~mask]
        splits.append(split)
    if len(args) == 1:
        return splits[0]
    else:
        return splits


def get_number_obs(obj):
    if hasattr(obj, "shape"):
        return obj.shape[0]
    elif hasattr(obj, "__len__"):
        return len(obj)
    else:
        raise ValueError("Could not get number observations. Obj has no attribute length or shape")
