from pystreed.base import BaseSTreeDSolver
from pystreed.binarizer import get_column_types
from pystreed.data import CSAData
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import Interval, StrOptions
from typing import Optional
import numpy as np
import pandas as pd
import numbers
from sklearn.preprocessing import StandardScaler

class STreeDCoxSurvivalAnalysis(BaseSTreeDSolver):
    _parameter_constraints: dict = {**BaseSTreeDSolver._parameter_constraints,
                                    "l1_ratio": [Interval(numbers.Real, 0, 1, closed="both")]
                                    }
    def __init__(self,
                 max_depth : int = 3,
                 max_num_nodes : Optional[int] = None,
                 min_leaf_node_size : int = 2,
                 time_limit : float = 5000,
                 hyper_tune : bool = False,
                 cost_complexity : float = 0.01,
                 use_branch_caching: bool = False,
                 use_dataset_caching: bool = True,
                 use_similarity_lower_bound: bool = True,
                 use_upper_bound: bool = True,
                 use_lower_bound: bool = True,
                 upper_bound: float = 2**31 - 1,
                 verbose : bool = False,
                 random_seed: int = 27,
                 continuous_binarize_strategy: str = 'quantile',
                 n_thresholds: int = 5,
                 n_categories: int = 5,
                 l1_ratio=0.99,
                 survival_validation="log-like"):
        """
        Construct a STreeDCoxSurvivalAnalysis

        Parameters:
            max_depth: the maximum depth of the tree
            max_num_nodes: the maximum number of branching nodes of the tree
            min_leaf_node_size: the minimum number of training instance that should end up in every leaf node
            time_limit: the time limit in seconds for fitting the tree
            hyper_tune: Use five-fold validation to tune the size of the tree to prevent overfitting
            cost_complexity: the cost of adding a branch node, expressed as a percentage. E.g., 0.01 means a branching node may be added if it increases the training accuracy by at least 1%.
            use_branch_caching: Enable/Disable branch caching (typically the slower caching strategy. May be faster in some scenario's)
            use_dataset_caching: Enable/Disable dataset caching (typically the faster caching strategy)
            use_similarity_lower_bound: Enable/Disable the similarity lower bound (Enabled typically results in a large runtime advantage)
            use_upper_bound: Enable/Disable the use of upper bounds (Enabled is typically faster)
            use_lower_bound: Enable/Disable the use of lower bounds (Enabled is typically faster)
            upper_bound: Search for a tree better than the provided upper bound
            verbose: Enable/Disable verbose output
            random_seed: the random seed used by the solver (for example when creating folds)
            continuous_binarization_strategy: the strategy used for binarizing continuous features
            n_thresholds: the number of thresholds to use per continuous feature
            n_categories: the number of categories to use per categorical feature
        """
        BaseSTreeDSolver.__init__(self, "cox-survival-analysis",
                                  max_depth=max_depth,
                                  max_num_nodes=max_num_nodes,
                                  min_leaf_node_size=min_leaf_node_size,
                                  time_limit=time_limit,
                                  cost_complexity=cost_complexity,
                                  feature_ordering="in-order",
                                  hyper_tune = hyper_tune,
                                  use_branch_caching=use_branch_caching,
                                  use_dataset_caching=use_dataset_caching,
                                  use_terminal_solver=False,
                                  use_similarity_lower_bound=use_similarity_lower_bound,
                                  use_upper_bound=use_upper_bound,
                                  use_lower_bound=use_lower_bound,
                                  upper_bound=upper_bound,
                                  verbose=verbose,
                                  random_seed=random_seed,
                                  continuous_binarize_strategy=continuous_binarize_strategy,
                                  n_thresholds=n_thresholds,
                                  n_categories=n_categories)
        self._label_type = np.double
        self.continuous_columns_ = None
        self.n_continuous_columns_ = 0
        self._reset_parameters.append("l1_ratio")
        self.l1_ratio = l1_ratio
        self.survival_validation = survival_validation

    def _initialize_param_handler(self):
        super()._initialize_param_handler()
        self._params.l1_ratio = self.l1_ratio
        self._params.survival_validation = self.survival_validation;
        if self.min_leaf_node_size < 5 * self.n_continuous_columns_:
            if self.verbose:
                print(f"Updating the minimum leaf node size to {5 * self.n_continuous_columns_}.")
        self._params.min_leaf_node_size = 5 * self.n_continuous_columns_
        return self._params

    def _process_extra_data(self, X, extra_data):
        return extra_data


    def sort_data(self, X, y):
        time = y[:, 0]

        return X, y

    def fit(self, X, y):
        X, y = self.sort_data(X, y)
        extra_data = []
        for i in range(len(y)):
            extra_data.append(CSAData(int(y[i][1]), list(X[i])))
        y = y[:, 0]
        self.n_continuous_columns_ = len(X[0])
        return super().fit(np.array(X), np.array(y), np.array(extra_data))

    def predict(self, X):
        check_is_fitted(self, "fit_result")
        extra_data = []
        for i in range(len(X)):
            extra_data.append(CSAData(0, list(X[i])))
        return super().predict(X, extra_data)

    def score(self, X, y):
        check_is_fitted(self, "fit_result")
        X, y = self.sort_data(X, y)
        extra_data = []
        for i in range(len(y)):
            extra_data.append(CSAData(int(y[i][1]), list(X[i])))
        return super().score(X, y[:, 0], extra_data)

