import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn import cross_validation
from sklearn.utils import check_array, as_float_array
from pyfm_fast import FM_fast, CSRDataset


LEARNING_RATE_TYPES = {"optimal": 0, "invscaling": 1, "constant": 2}
TASKS = {"regression": 0, "classification" : 1}


class BaseFM(BaseEstimator):
    """Factorization machine fitted by minimizing a regularized empirical loss with adaptive SGD.

    Parameters
    ----------

    num_factors : int
        The dimensionality of the factorized 2-way interactions
    num_iter : int
        Number of iterations
    k0 : bool
        Use bias. Defaults to true.
    k1 : bool
        Use 1-way interactions (learn feature weights).
        Defaults to true.
    init_stdev : double, optional
        Standard deviation for initialization of 2-way factors.
        Defaults to 0.01.
    validation_size : double, optional
        Proportion of the training set to use for validation.
        Defaults to 0.01.
    learning_rate_schedule : string, optional
        The learning rate:
            constant: eta = eta0
            optimal: eta = 1.0/(t+t0) [default]
            invscaling: eta = eta0 / pow(t, power_t)
    eta0 : double
        Defaults to 0.01
    power_t : double
        The exponent for inverse scaling learning rate [default 0.5].
    t0 : double
        Constant in the denominator for optimal learning rate schedule.
        Defaults to 0.001.
    task : string
        regression: Labels are real values.
        classification: Labels are either positive or negative.
    verbose : bool
        Whether or not to print current iteration, training error
    shuffle_training: bool
        Whether or not to shuffle training dataset before learning
    seed : int
        The seed of the pseudo random number generator
    """
    def __init__(self,
                 num_factors=10,
                 num_iter=1,
                 k0=True,
                 k1=True,
                 init_stdev=0.1,
                 validation_size=0.01,
                 learning_rate_schedule="optimal",
                 eta0=0.01,
                 power_t=0.5,
                 t0=0.001,
                 task='classification',
                 verbose=True,
                 shuffle_training=True,
                 seed = 28):

        self.num_factors = num_factors
        self.num_iter = num_iter
        self.sum = np.zeros(self.num_factors)
        self.sum_sqr = np.zeros(self.num_factors)
        self.k0 = k0
        self.k1 = k1
        self.init_stdev = init_stdev
        self.validation_size = validation_size
        self.task = task
        self.shuffle_training = shuffle_training
        self.seed = seed

        # Learning rate Parameters
        self.learning_rate_schedule = learning_rate_schedule
        self.eta0 = eta0
        self.power_t = power_t
        self.t0 = t0
        self.verbose = verbose

    def _validate_params(self):
        """Validate input params. """
        if not isinstance(self.shuffle_training, bool):
            raise ValueError("shuffle must be either True or False")
        if self.num_iter <= 0:
            raise ValueError("num_iter must be > zero")
        if self.learning_rate_schedule in ("constant", "invscaling"):
            if self.eta0 <= 0.0:
                raise ValueError("eta0 must be > 0")

        self.num_factors = int(self.num_factors)
        self.num_iter = int(self.num_iter)
        self.t0 = float(self.t0)
        self.power_t = float(self.power_t)
        try:
            self.eta0 = float(self.eta0)
        except TypeError:
            raise ValueError('eta0 expected float but got %r' % self.eta0)

    def _get_learning_rate_type(self, learning_rate):
        """Map learning rate string to int for cython"""
        try:
            return LEARNING_RATE_TYPES[learning_rate]
        except KeyError:
            raise ValueError("learning rate %s "
                             "is not supported. " % learning_rate)

    def _get_task(self, task):
        """Map task string to int for cython"""
        try:
            return TASKS[task]
        except KeyError:
            raise ValueError("task %s "
                             "is not supported. " % task)

    def _bool_to_int(self, bool_arg):
        """Map bool to int for cython"""
        if bool_arg == True:
            return 1
        else:
            return 0

    def _prepare_y(self,y):
        """Maps labels to [-1, 1] space"""
        y_i = np.ones(y.shape, dtype=np.float64, order="C")
        y_i[y != 1] = -1.0
        return y_i

    def fit(self, X, y):
        """Fit factorization machine using Stochastic Gradient Descent with Adaptive Regularization.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : numpy array of shape [n_samples]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        # X = as_float_array(X)
        # X, y = check_array(X, dtype=np.float64)
        if not isinstance(X, sp.csr_matrix):
            X = sp.csr_matrix(X)

        self._validate_params()

        self.t_ = 1.0
        self.max_target_ = y.max()
        self.min_target_ = y.min()

        # convert member variables to ints for use in cython
        k0 = self._bool_to_int(self.k0)
        k1 = self._bool_to_int(self.k1)
        shuffle_training = self._bool_to_int(self.shuffle_training)
        verbose = self._bool_to_int(self.verbose)
        learning_rate_schedule = self._get_learning_rate_type(self.learning_rate_schedule)
        task = self._get_task(self.task)

        # use sklearn to create a validation dataset for lambda updates
        if self.verbose:
            print("Creating validation dataset of %.2f of training for adaptive regularization"
                  % self.validation_size)
        X_train, validation, train_labels, validation_labels = cross_validation.train_test_split(
            X, y, test_size=self.validation_size, random_state=self.seed)

        self.n_features_ = X_train.shape[1]

        # Convert datasets to sklearn sequential datasets for fast traversal
        X_train_dataset = _make_dataset(X_train, train_labels)
        validation_dataset = _make_dataset(validation, validation_labels)

        # Set up params
        self.w0 = 0.0
        self.w = np.zeros(self.n_features_, dtype=np.float64)
        rng = np.random.RandomState(self.seed)
        self.v = rng.normal(scale=self.init_stdev,
                            size=(self.num_factors, self.n_features_)).astype(np.float64)

        self.fm_fast = FM_fast(self.w,
                               self.v,
                               self.num_factors,
                               self.n_features_,
                               self.num_iter,
                               k0,
                               k1,
                               self.w0,
                               self.t_,
                               self.t0,
                               self.power_t,
                               self.min_target_,
                               self.max_target_,
                               self.eta0,
                               learning_rate_schedule,
                               shuffle_training,
                               task,
                               self.seed,
                               verbose)

        self.fm_fast.fit(X_train_dataset, validation_dataset)
        return self

    def predict(self, X):
        """Predict using the factorization machine

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        float if X is one instance
        array, shape = [n_samples] if X is sparse matrix
           Predicted target values per element in X.
        """
        X, = check_arrays(X, sparse_format='csr', dtype=np.float64)
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        sparse_X = _make_dataset(X, np.ones(X.shape[0]))

        return self.fm_fast._predict(sparse_X)


class FMClassifier(BaseFM, ClassifierMixin):
    """Factorization machine fitted by minimizing a regularized empirical loss with adaptive SGD.

    Parameters
    ----------

    num_factors : int
        The dimensionality of the factorized 2-way interactions
    num_iter : int
        Number of iterations
    k0 : bool
        Use bias. Defaults to true.
    k1 : bool
        Use 1-way interactions (learn feature weights).
        Defaults to true.
    init_stdev : double, optional
        Standard deviation for initialization of 2-way factors.
        Defaults to 0.01.
    validation_size : double, optional
        Proportion of the training set to use for validation.
        Defaults to 0.01.
    learning_rate_schedule : string, optional
        The learning rate:
            constant: eta = eta0
            optimal: eta = 1.0/(t+t0) [default]
            invscaling: eta = eta0 / pow(t, power_t)
    eta0 : double
        Defaults to 0.01
    power_t : double
        The exponent for inverse scaling learning rate [default 0.5].
    t0 : double
        Constant in the denominator for optimal learning rate schedule.
        Defaults to 0.001.
    verbose : bool
        Whether or not to print current iteration, training error
    shuffle_training: bool
        Whether or not to shuffle training dataset before learning
    seed : int
        The seed of the pseudo random number generator
    """
    def __init__(self,
                 num_factors=10,
                 num_iter=1,
                 k0=True,
                 k1=True,
                 init_stdev=0.1,
                 validation_size=0.01,
                 learning_rate_schedule="optimal",
                 eta0=0.01,
                 power_t=0.5,
                 t0=0.001,
                 task='classification',
                 verbose=True,
                 shuffle_training=True,
                 seed = 28):
        super(FMClassifier, self).__init__(num_factors=num_factors,
                                           num_iter=num_iter,
                                           k0=k0, k1=k1,
                                           init_stdev=init_stdev,
                                           validation_size=validation_size,
                                           learning_rate_schedule=learning_rate_schedule,
                                           eta0=eta0,
                                           power_t=power_t,
                                           t0=t0,
                                           task='classification',
                                           verbose=verbose,
                                           shuffle_training=shuffle_training,
                                           seed=seed)

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError('FMClassifier only supports binary classification')

        y = (y * 2) - 1
        super(FMClassifier, self).fit(X, y)
        return self

    def predict(self, X):
        scores = super(FMClassifier, self).predict(X)
        pred = self.classes_[(scores >= 0.5).astype(np.int)]
        return pred

    def predict_proba(self, X):
        """Predict probabilities using the factorization machine

        Parameters
        ----------
        X : sparse matrix, shape = [n_samples, n_features]
        or
        X : single instance [1, n_features]

        Returns
        -------
        float if X is one instance
        array, shape = [n_samples] if X is sparse matrix
           Predicted target values per element in X.
        """
        proba = np.ones((X.shape[0], 2), dtype=np.float64)
        proba[:, 1] = super(FMClassifier, self).predict(X)
        proba[:, 0] -= proba[:, 1]
        return proba


class FMRegressor(BaseFM, RegressorMixin):
    """Factorization machine fitted by minimizing a regularized empirical loss with adaptive SGD.

    Parameters
    ----------

    num_factors : int
        The dimensionality of the factorized 2-way interactions
    num_iter : int
        Number of iterations
    k0 : bool
        Use bias. Defaults to true.
    k1 : bool
        Use 1-way interactions (learn feature weights).
        Defaults to true.
    init_stdev : double, optional
        Standard deviation for initialization of 2-way factors.
        Defaults to 0.01.
    validation_size : double, optional
        Proportion of the training set to use for validation.
        Defaults to 0.01.
    learning_rate_schedule : string, optional
        The learning rate:
            constant: eta = eta0
            optimal: eta = 1.0/(t+t0) [default]
            invscaling: eta = eta0 / pow(t, power_t)
    eta0 : double
        Defaults to 0.01
    power_t : double
        The exponent for inverse scaling learning rate [default 0.5].
    t0 : double
        Constant in the denominator for optimal learning rate schedule.
        Defaults to 0.001.
    verbose : bool
        Whether or not to print current iteration, training error
    shuffle_training: bool
        Whether or not to shuffle training dataset before learning
    seed : int
        The seed of the pseudo random number generator
    """
    def __init__(self,
                 num_factors=10,
                 num_iter=1,
                 k0=True,
                 k1=True,
                 init_stdev=0.1,
                 validation_size=0.01,
                 learning_rate_schedule="optimal",
                 eta0=0.01,
                 power_t=0.5,
                 t0=0.001,
                 verbose=True,
                 shuffle_training=True,
                 seed = 28):
        super(FMRegressor, self).__init__(
            num_factors=num_factors,
            num_iter=num_iter,
            k0=k0, k1=k1,
            init_stdev=init_stdev,
            validation_size=validation_size,
            learning_rate_schedule=learning_rate_schedule,
            eta0=eta0,
            power_t=power_t,
            t0=t0,
            task='regression',
            verbose=verbose,
            shuffle_training=shuffle_training,
            seed=seed)


def _make_dataset(X, y_i):
    """Create ``Dataset`` abstraction for sparse and dense inputs."""
    assert isinstance(X, sp.csr_matrix)
    assert X.dtype == np.float64

    # ignore sample weight for the moment
    sample_weight = np.ones(X.shape[0], dtype=np.float64, order='C')
    dataset = CSRDataset(X.data, X.indptr, X.indices, y_i, sample_weight)
    return dataset

