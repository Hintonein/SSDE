from matplotlib.pylab import f
import numpy as np
import scipy
import torch
from ssde.task import HierarchicalTask
from ssde.library import Library, Polynomial
from ssde.functions import create_tokens
from ssde.task.regression.polyfit import PolyOptimizer, make_poly_data
from ssde.pde import function_map


class RecursionTask(HierarchicalTask):
    """
    Class for the symbolic regression task. Discrete objects are expressions,
    which are evaluated based on their fitness to a specified dataset.
    """

    task_type = "recursion"

    def __init__(self,
                 function_set,
                 dataset,
                 metric="inv_nrmse",
                 metric_params=(1.0, ),
                 extra_metric_test=None,
                 extra_metric_test_params=(),
                 reward_noise=0.0,
                 reward_noise_type="r",
                 threshold=1e-12,
                 normalize_variance=False,
                 protected=False,
                 decision_tree_threshold_set=None,
                 poly_optimizer_params=None,
                 pde_forward=None,
                 start_n_var=1,
                 ablation=False):
        """
        Parameters
        ----------
        function_set : list or None
            List of allowable functions. If None, uses function_set according to
            benchmark dataset.

        dataset : dict, str, or tuple
            If dict: .dataset.BenchmarkDataset kwargs.
            If str ending with .csv: filename of dataset.
            If other str: name of benchmark dataset.
            If tuple: (X, y) data

        metric : str
            Name of reward function metric to use.

        metric_params : list
            List of metric-specific parameters.

        extra_metric_test : str
            Name of extra function metric to use for testing.

        extra_metric_test_params : list
            List of metric-specific parameters for extra test metric.

        reward_noise : float
            Noise level to use when computing reward.

        reward_noise_type : "y_hat" or "r"
            "y_hat" : N(0, reward_noise * y_rms_train) is added to y_hat values.
            "r" : N(0, reward_noise) is added to r.

        threshold : float
            Threshold of NMSE on noiseless data used to determine success.

        normalize_variance : bool
            If True and reward_noise_type=="r", reward is multiplied by
            1 / sqrt(1 + 12*reward_noise**2) (We assume r is U[0,1]).

        protected : bool
            Whether to use protected functions.

        decision_tree_threshold_set : list
            A set of constants {tj} for constructing nodes (xi < tj) in decision trees.

        poly_optimizer_params : dict
            Parameters for PolyOptimizer if poly token is in the library.
        """

        super(HierarchicalTask).__init__()
        """
        Configure (X, y) train/test data. There are four supported use cases:
        (1) named benchmark, (2) benchmark config, (3) filename, and (4) direct
        (X, y) data.
        """
        self.X_test = self.y_test = self.y_test_noiseless = None

        # Case 1: sklearn-like (X, y) data
        if isinstance(dataset, tuple):
            self.X_train = dataset[0]
            self.y_train = dataset[1]
            self.name = "solver"

        # Case 2: Dataset filename with npz
        elif isinstance(dataset, str) and dataset.endswith("npz"):
            data = np.load(dataset)
            self.X_train = [data['X'], data['X_bc']]
            self.y_train = [data['y'], data['y_bc']]
            self.name = "recursion_" + dataset.replace("/", "_")[:-4]

        # If not specified, set test data equal to the training data
        if self.X_test is None:
            self.X_test = self.X_train
            self.y_test = self.y_train
            self.y_test_noiseless = self.y_test


        # Set the number of input variables
        self.n_input_var = None
        # Set the sample size of parameters
        self.n_param = None
        self.start_n_var = start_n_var

        #  SSDE with Input as form of sklearn-like (X,Y) data.
        # X: List contains x and x_bc
        # Y: List contains y and y_bc
        self.X_u_train, self.X_bc_train = self.X_train
        self.y_u_train, self.y_bc_train = self.y_train
        self.X_u_test, self.X_bc_test = self.X_test
        self.y_u_test, self.y_bc_test = self.y_test
        self.y_u_test_noiseless, self.y_bc_test_noiseless = self.y_test_noiseless

        tokens = create_tokens(
            n_input_var=self.X_bc_train.shape[1],
            function_set=function_set,
            protected=protected,
            decision_tree_threshold_set=decision_tree_threshold_set,
            start_n_input=start_n_var)

        self.library = Library(tokens)
        self.n_param = self.y_bc_train.shape[1]

        if self.X_u_train is not None:
            self.n_input_var = self.y_u_train.shape[1]
            # Set the sample size of input variables
            self.n_input = self.X_u_train.shape[0]

            self.X_u_train_torch = torch.tensor(self.X_u_train, dtype=torch.float32, requires_grad=True)
            # Split the input data into two parts
            # the first part is used to calculate the pde loss
            # the second part is used to calculate the bc loss.
            self.y_train = [self.y_u_train[:self.n_input], self.y_u_train[self.n_input:]]
            self.y_train_torch = [torch.tensor(i, dtype=torch.float32) for i in self.y_train]
            # Just the pde_forward is a str or fucntion
            if isinstance(pde_forward, str):
                self.pde_forward = function_map[pde_forward]
            else:
                self.pde_forward = pde_forward

            # add the pde reward compute and metric
            self.const_pdeopt_metric, _, _ = make_regression_metric("neg_smse", self.y_train)
            self.const_torchopt_metric, _, _ = make_regression_metric(
                "neg_smse_torch", self.y_train)
            self.pdemetric, _, _ = make_regression_metric(
                "inv_mrmse", self.y_train, *metric_params)
        else:
            # add the bc compute and metric
            self.y_train = self.y_bc_train
            self.y_test = self.y_bc_test
            self.y_test_noiseless = self.y_test

        # Use mse of y_bc as the metric for initial const optimization
        self.const_opt_metric, _, _ = make_regression_metric(
            "neg_mse", self.y_bc_train)
        """
        Configure ablation flag
        # If ablation is True, the constants are directly optimized based on pde constraints
        """
        self.ablation = ablation
        self.stochastic = reward_noise > 0.0
        """
        Configure train/test reward metrics.
        """
        self.threshold = threshold
        print('The threshold of evaluation is:', self.threshold)
        self.metric, self.invalid_reward, self.max_reward = make_regression_metric(
            metric, self.y_train, *metric_params)
        self.extra_metric_test = extra_metric_test
        if extra_metric_test is not None:
            self.metric_test, _, _ = make_regression_metric(
                extra_metric_test, self.y_test, *extra_metric_test_params)
        else:
            self.metric_test = None


    # pde 过程的前向计算
    def intermediate_forward(self, y, X):
        try:
            f = [self.pde_forward(y[:, i:i + 1], X) for i in range(y.shape[1])]
            f = torch.cat(f, dim=1)
            # f as the constrain of the pde
            # y as the constrain of bcs
            y_hat = [f, y]
        except RuntimeError:
            y_hat = [None, None]
        return y_hat

    def reward_function(self, p, optimizing=None, dt=True):
        if optimizing is not None:
            if optimizing:
                # reward for pre constant optimization(without pde error)
                # Note: y_bc_hat.shape[1] is the number of parameters
                y_bc_hat = p.execute(self.X_bc_train)
                assert y_bc_hat.shape[1] == self.n_param, \
                    f"y_bc_hat.shape[1] is {y_bc_hat.shape[1]}, but self.n_param is {self.n_param}"
                if p.invalid:
                    return -1.0
                return self.const_opt_metric(self.y_bc_train, y_bc_hat)
            else:
                # reward for the re constant optimization(with pde error and bc error)
                if p.invalid:
                    # Set for torch optimization when the process is wrong.
                    return torch.tensor([1.0])
                y = p.execute(self.X_u_train_torch)
                y_hat = self.intermediate_forward(y,
                                                  self.X_u_train_torch)

                if y_hat[0] is None or not all(
                    [torch.isnan(i).any() == False for i in y_hat]):
                    p.invalid = True
                    return torch.tensor([1.0])
                return self.const_torchopt_metric(self.y_train_torch, y_hat)

        if p.invalid:
            return self.invalid_reward
        if self.X_u_train is not None and dt:
            y = p.execute(self.X_u_train_torch)
            y_hat = self.intermediate_forward(y, self.X_u_train_torch)
            if y_hat[0] is None or not all(
                [torch.isnan(i).any() == False for i in y_hat]):
                p.invalid = True
                return self.invalid_reward
            if p.bc_mse is not None:
                    y_hat = [i.detach().numpy() for i in y_hat]
                    r = 1/(1 + (sum([np.sqrt(np.mean((self.y_train[i] - y_hat[i])**2)) for i in range(len(y_hat))]) + np.sqrt(p.bc_mse))/(len(y_hat)+1))
            else:
                r = self.pdemetric(self.y_train,
                                [i.detach().numpy() for i in y_hat])
        else:
            y_hat = p.execute(self.X_bc_train)
            if p.invalid or np.isnan(y_hat).any():
                return self.invalid_reward
            r = self.metric(self.y_bc_train, y_hat)

        return r

    def evaluate(self, p):
        # Including two cases, one is that x_u_train is an empty set, indicating that the intermediate variables are optimized
        # The other is that x_u_train is not an empty set, indicating that the evaluation of the first variable
        flag = 0
        if self.n_param > 1:
            if self.X_u_train is not None:
                try:
                    y = p.execute(self.X_u_train_torch)
                    y_bc_hat1 = self.intermediate_forward(y, self.X_u_train_torch)
                    y_bc_hat1 =  [i.detach().numpy() for i in y_bc_hat1]
                    if y_bc_hat1[0].shape[1] != self.n_input_var:
                        flag = 1
                        y_bc_hat2 = p.execute(self.X_bc_train)
                except:
                    flag = 1
                    y_bc_hat2 = p.execute(self.X_bc_train)
            else:
                flag = 1
                y_bc_hat2 = p.execute(self.X_bc_train)

            if p.invalid:
                mse_test = None
                mse_test_noiseless = None
                success = False
            else:
                if flag:
                    mse_test =  np.mean((self.y_bc_train - y_bc_hat2) ** 2)
                elif p.bc_mse != -1:
                    mse_test = (sum([np.mean((self.y_train[i] - y_bc_hat1[i])**2) for i in range(len(y_bc_hat1))]) 
                                + p.bc_mse)/(len(y_bc_hat1)+1)
                mse_test_noiseless = mse_test
                success = mse_test_noiseless < self.threshold
            info = {
                "mse_test" : mse_test,
                "mse_test_noiseless" : mse_test_noiseless,
                "success" : success
            }
        else:
            y_hat = p.execute(self.X_bc_train)
            if y_hat[0] is None:
                p.invalid = True
            if p.invalid:
                mse_test = None
                mse_test_noiseless = None
                success = False
            else:
                temp = [
                    np.mean((self.y_test[i] - y_hat[i])**2)
                    for i in range(len(self.y_test))
                ]
                mse_test = sum(temp) / len(self.y_test)
                mse_test_noiseless = sum([
                    np.mean((self.y_test_noiseless[i] - y_hat[i])**2)
                    for i in range(len(self.y_test))
                ]) / len(self.y_test)
                success = mse_test_noiseless < self.threshold
            info = {
                "mse_test": mse_test,
                "mse_test_noiseless": mse_test_noiseless,
                "success": success
            }

        if self.metric_test is not None:
            if p.invalid:
                m_test = None
                m_test_noiseless = None
            else:
                m_test = self.metric_test(self.y_test, y_hat)
                m_test_noiseless = self.metric_test(self.y_test_noiseless,
                                                    y_hat)

            info.update({
                self.extra_metric_test: m_test,
                self.extra_metric_test + '_noiseless': m_test_noiseless
            })

        return info


def make_regression_metric(name, y_train, *args):
    """
    Factory function for a regression metric. This includes a closures for
    metric parameters and the variance of the training data.

    Parameters
    ----------

    name : str
        Name of metric. See all_metrics for supported metrics.

    args : args
        Metric-specific parameters

    Returns
    -------

    metric : function
        Regression metric mapping true and estimated values to a scalar.

    invalid_reward: float or None
        Reward value to use for invalid expression. If None, the training
        algorithm must handle it, e.g. by rejecting the sample.

    max_reward: float
        Maximum possible reward under this metric.
    """
    if not isinstance(y_train, list):
        var_y = np.var(y_train)
    else:
        var_y = np.sum([np.var(y) for y in y_train])

    all_metrics = {

        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        "neg_mse": (lambda y, y_hat: -np.mean((y - y_hat)**2), 0),

        # Negative sum of mean squared error of pdes
        # Range: [-inf, 0]
        "neg_smse": (lambda y, y_hat: -sum(
            [np.mean((y[i] - y_hat[i])**2) for i in range(len(y))]), 0),

        # Negative mean squared error of torch version
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        "neg_smse_torch": (lambda y, y_hat: -sum(
            [torch.mean((y[i] - y_hat[i])**2) for i in range(len(y))]), 0),

        # Negative root mean squared error
        # Range: [-inf, 0]
        # Value = -sqrt(var(y)) when y_hat == mean(y)
        "neg_rmse": (lambda y, y_hat: -np.sqrt(np.mean((y - y_hat)**2)), 0),

        # Negative normalized mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nmse": (lambda y, y_hat: -np.mean((y - y_hat)**2) / var_y, 0),

        # Negative normalized root mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nrmse":
        (lambda y, y_hat: -np.sqrt(np.mean((y - y_hat)**2) / var_y), 0),

        # (Protected) negative log mean squared error
        # Range: [-inf, 0]
        # Value = -log(1 + var(y)) when y_hat == mean(y)
        "neglog_mse": (lambda y, y_hat: -np.log(1 + np.mean(
            (y - y_hat)**2)), 0),

        # (Protected) inverse mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]*var(y)) when y_hat == mean(y)
        "inv_mse": (lambda y, y_hat: 1 / (1 + args[0] * np.mean(
            (y - y_hat)**2)), 1),

        # (Protected) inverse mean squared error of pdes
        # Range: [0, 1]
        "inv_mrmse": (lambda y, y_hat: 1 / (1 + args[0] * sum(
            [np.sqrt(np.mean((y[i] - y_hat[i])**2))
             for i in range(len(y))]) / len(y)), 1),

        # (Protected) inverse mean squared error of torch version
        # Range: [0, 1]
        # Value = 1/(1 + args[0]*var(y)) when y_hat == mean(y)
        "inv_mse_torch": (lambda y, y_hat: 1 / (1 + args[0] * torch.mean(
            (y - y_hat)**2)), 1),

        # (Protected) inverse normalized mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nmse": (lambda y, y_hat: 1 / (1 + args[0] * np.mean(
            (y - y_hat)**2) / var_y), 1),

        # (Protected) inverse normalized root mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nrmse": (lambda y, y_hat: 1 /
                      (1 + args[0] * np.sqrt(np.mean(
                          (y - y_hat)**2) / var_y)), 1),

        # Fraction of predicted points within p0*abs(y) + p1 band of the true value
        # Range: [0, 1]
        "fraction":
        (lambda y, y_hat: np.mean(abs(y - y_hat) < args[0] * abs(y) + args[1]),
         2),

        # Pearson correlation coefficient
        # Range: [0, 1]
        "pearson": (lambda y, y_hat: scipy.stats.pearsonr(y, y_hat)[0], 0),

        # Spearman correlation coefficient
        # Range: [0, 1]
        "spearman": (lambda y, y_hat: scipy.stats.spearmanr(y, y_hat)[0], 0)
    }

    assert name in all_metrics, "Unrecognized reward function name."
    assert len(args) == all_metrics[name][
        1], "For {}, expected {} reward function parameters; received {}.".format(
            name, all_metrics[name][1], len(args))
    metric = all_metrics[name][0]

    # For negative MSE-based rewards, invalid reward is the value of the reward function when y_hat = mean(y)
    # For inverse MSE-based rewards, invalid reward is 0.0
    # For non-MSE-based rewards, invalid reward is the minimum value of the reward function's range
    all_invalid_rewards = {
        "neg_mse": -var_y,
        "neg_smse": -var_y,
        "neg_smse_torch": -var_y,
        "neg_rmse": -np.sqrt(var_y),
        "neg_nmse": -1.0,
        "neg_nrmse": -1.0,
        "neglog_mse": -np.log(1 + var_y),
        "inv_mse": 0.0,  #1/(1 + args[0]*var_y),
        "inv_mrmse": 0.0,  #1/(1 + args[0]*var_y),
        "inv_mse_torch": 0.0,  #1/(1 + args[0]*var_y),
        "inv_nmse": 0.0,  #1/(1 + args[0]),
        "inv_nrmse": 0.0,  #1/(1 + args[0]),
        "fraction": 0.0,
        "pearson": 0.0,
        "spearman": 0.0
    }
    invalid_reward = all_invalid_rewards[name]

    all_max_rewards = {
        "neg_mse": 0.0,
        "neg_smse": 0.0,
        "neg_smse_torch": 0.0,
        "neg_rmse": 0.0,
        "neg_nmse": 0.0,
        "neg_nrmse": 0.0,
        "neglog_mse": 0.0,
        "inv_mse": 1.0,
        "inv_mrmse": 1.0,
        "inv_mse_torch": 1.0,
        "inv_nmse": 1.0,
        "inv_nrmse": 1.0,
        "fraction": 1.0,
        "pearson": 1.0,
        "spearman": 1.0
    }
    max_reward = all_max_rewards[name]

    return metric, invalid_reward, max_reward
