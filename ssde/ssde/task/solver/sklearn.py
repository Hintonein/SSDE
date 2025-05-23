from copy import deepcopy

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ssde import DeepSymbolicOptimizer


class PDESymbolicSolver(DeepSymbolicOptimizer, BaseEstimator, RegressorMixin):
    """
    Sklearn interface for deep symbolic regression.
    """

    def __init__(self, config=None):
        if config is None:
            config = {"task": {"task_type": "solver"}}
        DeepSymbolicOptimizer.__init__(self, config)

    def fit(self, X_input, y_input, start_n_var=1, debuglist=None, diff=None):
        # reg_expr: regressed expression by last fit

        # Update the Task
        config = deepcopy(self.config)
        config["task"]["dataset"] = (X_input, y_input)
        config["task"]["start_n_var"] = start_n_var
        if diff is not None:
            config["task"]["pde_forward"] = diff
        # Turn off file saving
        config["experiment"]["logdir"] = None

        # # TBD: Add support for gp-meld and sklearn interface. Currently, gp-meld
        # # relies on BenchmarkDataset objects, not (X, y) data.
        # if config["gp_meld"].get("run_gp_meld"):
        #     print("WARNING: GP-meld not yet supported for sklearn interface.")
        # config["gp_meld"]["run_gp_meld"] = False

        self.set_config(config)
        if debuglist is not None:
            train_result, debug = self.train(debuglist)
        else:
            train_result = self.train(debuglist)
        self.program_ = train_result["program"]
        self.train_result_ = train_result

        return self

    def predict(self, X):

        check_is_fitted(self, "program_")

        return self.program_.execute(X)
