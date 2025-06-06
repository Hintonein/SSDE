"""Utility functions used in deep symbolic optimization."""

import collections
import copy
import functools
import numpy as np
import time
import importlib
import random
import re
import os
import sys
import pandas as pd
import logging
import sympy.parsing.sympy_parser as sympy_parser
import sympy

from typing import Callable
from functools import wraps

def preserve_global_rng_state(f: Callable):
    """
    Decorator that saves the internal state of the global random number
    generator before call to function and sets it back to that state
    after the call

    Parameters
    ----------
    f : Callable
        Function to decorate

    Returns
    _______
    Callable
        Decorated function that saves global random state and resets to it after
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        rng_state = random.getstate()
        result = f(*args, **kwargs)
        random.setstate(rng_state)
        return result
    return decorated


# We wrap the sympy functions with preserve_global_rng_state
# as the sympy routines appear to non-deterministically
# re-seed the global random generator which can influence GP results.
# This problem seems to be resolved in sympy in commit
# https://github.com/sympy/sympy/pull/22433
# These functions should be used instead of the sympy functions directly
pretty = preserve_global_rng_state(sympy.pretty)
parse_expr = preserve_global_rng_state(sympy_parser.parse_expr)


def is_float(s):
    """Determine whether the input variable can be cast to float."""

    try:
        float(s)
        return True
    except ValueError:
        return False


# Adapted from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points given an array of costs.

    Parameters
    ----------

    costs : np.ndarray
        Array of shape (n_points, n_costs).

    Returns
    -------

    is_efficient_maek : np.ndarray (dtype:bool)
        Array of which elements in costs are pareto-efficient.
    """

    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


class cached_property(object):
    """
    Decorator used for lazy evaluation of an object attribute. The property
    should be non-mutable, since it replaces itself.
    """

    def __init__(self, getter):
        self.getter = getter

        functools.update_wrapper(self, getter)

    def __get__(self, obj, cls):
        if obj is None:
            return self

        value = self.getter(obj)
        setattr(obj, self.getter.__name__, value)
        return value


def weighted_quantile(values, weights, q):
    """
    Computes the weighted quantile, equivalent to the exact quantile of the
    empirical distribution.

    Given ordered samples x_1 <= ... <= x_n, with corresponding weights w_1,
    ..., w_n, where sum_i(w_i) = 1.0, the weighted quantile is the minimum x_i
    for which the cumulative sum up to x_i is greater than or equal to 1.

    Quantile = min{ x_i | x_1 + ... + x_i >= q }
    """

    sorted_indices = np.argsort(values)
    sorted_weights = weights[sorted_indices]
    sorted_values = values[sorted_indices]
    cum_sorted_weights = np.cumsum(sorted_weights)
    i_quantile = np.argmax(cum_sorted_weights >= q)
    quantile = sorted_values[i_quantile]

    # NOTE: This implementation is equivalent to (but much faster than) the
    # following:
    # from scipy import stats
    # empirical_dist = stats.rv_discrete(name='empirical_dist', values=(values, weights))
    # quantile = empirical_dist.ppf(q)

    return quantile


# Entropy computation in batch
def empirical_entropy(labels):

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.
    # Compute entropy
    for i in probs:
        ent -= i * np.log(i)

    return np.array(ent, dtype=np.float32)


def get_duration(start_time):
    return get_human_readable_time(time.time() - start_time)


def get_human_readable_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return "{:02d}:{:02d}:{:02d}:{:05.2f}".format(int(d), int(h), int(m), s)


def safe_merge_dicts(base_dict, update_dict):
    """Merges two dictionaries without changing the source dictionaries.

    Parameters
    ----------
        base_dict : dict
            Source dictionary with initial values.
        update_dict : dict
            Dictionary with changed values to update the base dictionary.

    Returns
    -------
        new_dict : dict
            Dictionary containing values from the merged dictionaries.
    """
    if base_dict is None:
        return update_dict
    base_dict = copy.deepcopy(base_dict)
    for key, value in update_dict.items():
        if isinstance(value, collections.Mapping):
            base_dict[key] = safe_merge_dicts(base_dict.get(key, {}), value)
        else:
            base_dict[key] = value
    return base_dict


def safe_update_summary(csv_path, new_data):
    """Updates a summary csv file with new rows. Adds new columns
    in existing data if necessary. New rows are distinguished by
    the run seed.

    Parameters
    ----------
        csv_path : str
            String with the path to the csv file.
        new_data : dict
            Dictionary containing values to be saved in the csv file.

    Returns
    -------
        bool
            Boolean value to indicate if saving the data to file worked.
    """
    try:
        new_data_pd = pd.DataFrame(new_data, index=[0])
        new_data_pd.set_index('seed', inplace=True)
        if os.path.isfile(csv_path):
            old_data_pd = pd.read_csv(csv_path)
            old_data_pd.set_index('seed', inplace=True)
            merged_df = pd.concat([old_data_pd, new_data_pd], axis=0, ignore_index=False)
            merged_df.to_csv(csv_path, header=True, mode='w+', index=True)
        else:
            new_data_pd.to_csv(csv_path, header=True, mode='w+', index=True)
        return True
    except:
        return False


def import_custom_source(import_source):
    """
    Provides a way to import custom modules. The return will be a reference to the desired source
    Parameters
    ----------
        import_source : import path
            Source to import from, for most purposes: <module_name>:<class or function name>

    Returns
    -------
        mod : ref
            reference to the imported module
    """

    # Partially validates if the import_source is in correct format
    regex = '[\w._]+:[\w._]+' #lib_name:class_name
    m = re.match(pattern=regex, string=import_source)
    # Partial matches mean that the import will fail
    assert m is not None and m.end() == len(import_source), "*** Failed to import malformed source string: "+import_source

    source, type = import_source.split(':')

    # Dynamically imports the configured source
    mod = importlib.import_module(source)
    func = getattr(mod, type)

    return func

def pad_action_obs_priors(actions, obs, priors, pad_length):
    """
    Will pad action, obs, priors with zeros.  
    
    Parameters
    ----------
        actions : np array
            Standard actions array of tokens
        obs : np array
            Standard observations array
        priors : np array
            Standard priors array
        pdd_length : int

    Returns
    -------
        actions : np array
            Standard actions array of tokens padded with zeros at the end columns
        obs : np array
            Standard observations array padded with zeros at the end columns
        priors : np array
            Standard priors array padded with zeros at the end columns
    """
    assert isinstance(pad_length,int)
    assert pad_length >= 0
    
    actions = np.pad(actions, ((0,0),(0,pad_length)), 'constant', constant_values=((0,0),(0,0)))
    obs = [ np.pad(o, ((0,0),(0,pad_length)), 'constant', constant_values=((0,0),(0,0))) for o in obs ]
    priors = np.pad(priors, ((0,0),(0,pad_length),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))

    return actions, obs, priors


def make_batch_ph(name : str, n_choices : int):
    """
    Generates dictionary containing placeholders needed for a batch of sequences.
    
    Parameters
    ----------
        names : str
            Name of tensorflow scope for this batch

        n_choices : int
            Number of choices in priors

    Returns
    -------
        batch_ph : dict
            Dictionary of placeholders
    """

    # Lazy import
    import tensorflow as tf
    from ssde.memory import Batch
    from ssde.program import Program

    with tf.name_scope(name):
        batch_ph = {
            "actions": tf.placeholder(tf.int32, [None, None]),
            "obs": tf.placeholder(tf.float32, [None, Program.task.OBS_DIM, None]),
            "priors": tf.placeholder(tf.float32, [None, None, n_choices]),
            "lengths": tf.placeholder(tf.int32, [None, ]),
            "rewards": tf.placeholder(tf.float32, [None], name="r"),
            "on_policy": tf.placeholder(tf.int32, [None, ])
         }
        batch_ph = Batch(**batch_ph)
    return batch_ph

def rect(xmin, xmax, n, method="uniform"):
    """
    Args:
        xmin: Coordinate of bottom left corner.
        xmax: Coordinate of top right corner.
        n: Number of points to sample.
        method: Sampling method. Either "uniform" or "pseudo".
    """
    xmax = np.array(xmax)
    xmin = np.array(xmin)
    perimeter = 2 * np.sum(xmax - xmin)
    if method == "uniform":
        nx, ny = np.ceil(n / perimeter * (xmax - xmin)).astype(int)
        xbot = np.hstack((
            np.linspace(xmin[0], xmax[0], num=nx, endpoint=False)[:, None],
            np.full([nx, 1], xmin[1]),
        ))
        yrig = np.hstack((
            np.full([ny, 1], xmax[0]),
            np.linspace(xmin[1], xmax[1], num=ny, endpoint=False)[:, None],
        ))
        xtop = np.hstack((
            np.linspace(xmin[0], xmax[0], num=nx + 1)[1:, None],
            np.full([nx, 1], xmax[1]),
        ))
        ylef = np.hstack((
            np.full([ny, 1], xmin[0]),
            np.linspace(xmin[1], xmax[1], num=ny + 1)[1:, None],
        ))
        x = np.vstack((xbot, yrig, xtop, ylef))
        if n != len(x):
            print("Warning: {} points required, but {} points sampled.".format(
                n, len(x)))
        return x
    elif method == "random":
        nx, ny = np.ceil(n / perimeter * (xmax - xmin)).astype(int)
        xbot = np.hstack((
            np.random.uniform(xmin[0], xmax[0], nx)[:, None],
            np.full([nx, 1], xmin[1]),
        ))
        yrig = np.hstack((
            np.full([ny, 1], xmax[0]),
            np.random.uniform(xmin[1], xmax[1], ny)[:, None],
        ))
        xtop = np.hstack((
            np.random.uniform(xmin[0], xmax[0], nx)[:, None],
            np.full([nx, 1], xmax[1]),
        ))
        ylef = np.hstack((
            np.full([ny, 1], xmin[0]),
            np.random.uniform(xmin[1], xmax[1], ny)[:, None],
        ))
        x = np.vstack((xbot, yrig, xtop, ylef))
        if n != len(x):
            print("Warning: {} points required, but {} points sampled.".format(
                n, len(x)))
        return x

    elif method == "pseudo":
        # Randomly sample points on the perimeter
        l1 = xmax[0] - xmin[0]
        l2 = l1 + xmax[1] - xmin[1]
        l3 = l2 + l1
        u = np.ravel(np.random.random(size=(n + 2, 1)))
        # Remove the possible points very close to the corners
        u = u[np.logical_not(np.isclose(u, l1 / perimeter))]
        u = u[np.logical_not(np.isclose(u, l3 / perimeter))]
        u = u[:n]
        u *= perimeter
        x = []
        for lateral in u:
            if lateral < l1:
                x.append([xmin[0] + lateral, xmin[1]])
            elif lateral < l2:
                x.append([xmax[0], xmin[1] + lateral - l1])
            elif lateral < l3:
                x.append([xmax[0] - lateral + l2, xmax[1]])
            else:
                x.append([xmin[0], xmax[1] - lateral + l3])
        return np.vstack(x)


def cube(left, right, num_samples, method='uniform'):
    """
    Sample points on [left, right]^3 
    
    Parameters
    ----------
    left: min value of spatio
    right: max value of spatio
    num_samples: number of sampled points
    method: 'uniform' 或 'grid'
    
    
    return: ndarrya (N, 3)
    """
    if method == 'uniform':
        points = []
        for _ in range(num_samples):
            face = np.random.choice(6)  # 选择一个面
            x = np.random.uniform(left, right)
            y = np.random.uniform(left, right)
            z = np.random.uniform(left, right)

            if face == 0:
                points.append((left, x, y))
            elif face == 1:
                points.append((right, x, y))
            elif face == 2:
                points.append((x, left, y))
            elif face == 3:
                points.append((x, right, y))
            elif face == 4:
                points.append((x, y, left))
            else:
                points.append((x, y, right))

        return np.array(points)

    elif method == 'grid':
        coords = np.linspace(left, right, num_samples)
        grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
        is_boundary = (np.isclose(grid[:, 0], left) | np.isclose(grid[:, 0], right) |
                       np.isclose(grid[:, 1], left) | np.isclose(grid[:, 1], right) |
                       np.isclose(grid[:, 2], left) | np.isclose(grid[:, 2], right))
        boundary_points = grid[is_boundary]
        return boundary_points

    else:
        raise ValueError("Unsupported sampling method: choose 'uniform' or 'grid'.")



def jupyter_logging(default_file_path):
    """
    Decorator to redirect output into a specified file, allowing dynamic path changes during calls.

    Parameters
    ----------
    default_file_path: str
        Default log file path if not specified during function call.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract log_path from kwargs, use default if not provided
            current_log_path = kwargs.pop('log_path', default_file_path)

            # Create a new logger instance
            logger = logging.getLogger(func.__name__)  # Use function name as logger name
            logger.setLevel(logging.INFO)

            # Remove existing handlers to avoid duplicate logs
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Add a file handler for the specified file
            file_handler = logging.FileHandler(current_log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
            
            class RedirectOutput:
                def __init__(self, logger):
                    self.logger = logger

                def write(self, message):
                    if message.strip() != "":
                        self.logger.info(message)

                def flush(self):
                    pass

            # Save the original stdout and stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            # Redirect stdout and stderr
            sys.stdout = RedirectOutput(logger)
            sys.stderr = RedirectOutput(logger)

            try:
                # Execute the decorated function
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error occurred: {str(e)}")
                raise  # Re-raise the exception
            finally:
                # Restore the original stdout and stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            return result
        return wrapper
    return decorator


def get_stboundary(num_ic, num_bc, dim, left, right, t_start = 0.0, t_end = 1.0):
    total_dim = dim 
    dim = total_dim - 1
    num_pts = num_ic + num_bc
    pts = np.zeros((num_pts, total_dim))
    
    space_left = left
    space_right = right
    
    pts[:num_ic, 0] = t_start  
    pts[:num_ic, 1:] = np.random.rand(num_ic, dim) * (space_right - space_left) + space_left
    

    pts[num_ic:, 0] = np.random.rand(num_bc) * (t_end - t_start) + t_start 
    pts[num_ic:, 1:] = np.random.rand(num_bc, dim) * (space_right - space_left) + space_left   
    
    for i in range(num_ic, num_pts):
        fixed_dim = np.random.randint(1, dim+1, (1,)).item()
        if np.random.rand(1) < 0.5:
            pts[i, fixed_dim] = space_left
        else:
            pts[i, fixed_dim] = space_right
    
    return pts