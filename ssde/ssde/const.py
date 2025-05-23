"""Constant optimizer used for deep symbolic optimization."""

from functools import partial

import numpy as np
from scipy.optimize import minimize, least_squares
import torch


def make_const_optimizer(name, **kwargs):
    """Returns a ConstOptimizer given a name and keyword arguments"""

    const_optimizers = {
        None : Dummy,
        "dummy" : Dummy,
        "scipy" : ScipyMinimize,
        "torch" : PytorchMinimize,
        "lm"    : ScipyLeastSquares,
    }

    return const_optimizers[name](**kwargs)

def torch_minimize(f,x,method='L-BFGS', options=None, inner_iter=5, tol=1e-7):
    """
    Optimizes an objective function from an initial guess.

    The objective function is the negative of the base reward (reward
    without penalty) used for training. Optimization excludes any penalties
    because they are constant w.r.t. to the constants being optimized.

    Parameters
    ----------
    f : function mapping torch.tensor to float
        Objective function (negative base reward).

    x : list of torch.tensor
        Initial guess for constant placeholders.

    Returns
    -------
    x : np.ndarray
        Vector of optimized constants.
    """
    # x = torch.tensor(x0, requires_grad=True)
    optimizer = torch.optim.LBFGS(x, lr=1, max_iter=20, tolerance_grad=tol)
    class InvaildException(Exception):
        pass
    def closure():
        optimizer.zero_grad()
        loss = f(x)
        if loss != -1:
            try:
                loss.backward()
            except AttributeError:
                print(f'loss is not a tensor:{loss}')
                raise InvaildException
        else:
            # expression is not valid
            raise InvaildException
        return loss
    for i in range(inner_iter):
        try:
            optimizer.step(closure)
        except InvaildException:
            break
    # assert len(x) == 1 
    return {'x': [i.detach().numpy().astype(np.float32) for i in x], 'fun': f(x).detach().numpy()}

class ConstOptimizer(object):
    """Base class for constant optimizer"""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs


    def __call__(self, f, x0):
        """
        Optimizes an objective function from an initial guess.

        The objective function is the negative of the base reward (reward
        without penalty) used for training. Optimization excludes any penalties
        because they are constant w.r.t. to the constants being optimized.

        Parameters
        ----------
        f : function mapping np.ndarray to float
            Objective function (negative base reward).

        x0 : np.ndarray
            Initial guess for constant placeholders.

        Returns
        -------
        x : np.ndarray
            Vector of optimized constants.
        """
        raise NotImplementedError


class Dummy(ConstOptimizer):
    """Dummy class that selects the initial guess for each constant"""

    def __init__(self, **kwargs):
        super(Dummy, self).__init__(**kwargs)

    
    def __call__(self, f, x0):
        return x0
        

class ScipyMinimize(ConstOptimizer):
    """SciPy's non-linear optimizer"""

    def __init__(self, **kwargs):
        super(ScipyMinimize, self).__init__(**kwargs)

    
    def __call__(self, f, x0):
        with np.errstate(divide='ignore'):
            opt_result = partial(minimize, **self.kwargs)(f, x0)
        x = opt_result['x']
        func = opt_result['fun']
        return x, func
    
class PytorchMinimize(ConstOptimizer):
    """Pytorch's non-linear optimizer"""

    def __init__(self, **kwargs):
        super(PytorchMinimize, self).__init__(**kwargs)

    
    def __call__(self, f, x0):
        with np.errstate(divide='ignore'):
            opt_result = partial(torch_minimize, **self.kwargs)(f, x0)
        x = opt_result['x']
        func = opt_result['fun']
        return x, func

class ScipyLeastSquares(ConstOptimizer):
    """SciPy's non-linear optimizer"""

    def __init__(self, **kwargs):
        super(ScipyLeastSquares, self).__init__(**kwargs)

    
    def __call__(self, f, x0):
        with np.errstate(divide='ignore'):
            opt_result = partial(least_squares, **self.kwargs)(f, x0)
        x = opt_result['x']
        func = opt_result['fun']
        return x, func