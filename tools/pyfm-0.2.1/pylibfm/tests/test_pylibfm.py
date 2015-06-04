import numpy as np
import scipy.sparse as sp
import cPickle as pickle

from pylibfm import FMRegressor
from pylibfm import FMClassifier

rng = np.random.RandomState(13)
n = 1000
X = rng.rand(n, 100)
y = rng.randint(0, 2, n)


def test_smoke_classification():
    for X_ in (X, sp.csc_matrix(X), sp.csr_matrix(X), sp.coo_matrix(X)):
        for lr in ['optimal', 'invscaling', 'constant']:
            est = FMClassifier(verbose=0, seed=13, num_iter=2, learning_rate_schedule=lr).fit(X_, y)
            proba = est.predict_proba(X_)
            assert proba.shape == (n, 2)
            np.testing.assert_array_almost_equal(proba[:3], np.array([[0.45637266, 0.54362734],
                                                                      [0.47387578, 0.52612422],
                                                                      [0.48227419, 0.51772581]]))

            pred = est.predict(X_)
            assert pred.shape == (n, )
            np.testing.assert_array_almost_equal(pred[:5], np.ones(5))


def test_clf_single():
    est = FMClassifier(verbose=0, seed=13, num_iter=2).fit(X, y)
    out = est.predict(X[:1])
    assert isinstance(out, np.ndarray)
    out = est.predict_proba(X[:1])
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 2)


def test_smoke_regression():
    for X_ in (X, sp.csc_matrix(X), sp.csr_matrix(X), sp.coo_matrix(X)):
        for lr in ['optimal', 'invscaling', 'constant']:
            est = FMRegressor(verbose=0, seed=13, num_iter=2, learning_rate_schedule=lr).fit(X_, y)
            pred = est.predict(X_)
            assert pred.shape == (n, )
            np.testing.assert_array_almost_equal(pred[:5],
                                                 np.array([1.0, 0.35404291, 0.57543318,
                                                           0.53575063, 0.98801102]))


def test_pickle():
    est = FMClassifier(verbose=0, seed=13, num_iter=2).fit(X, y)
    out = est.predict(X)
    ser = pickle.dumps(est)

    est2 = pickle.loads(ser)
    out2 = est2.predict(X)
    np.testing.assert_array_equal(out, out2)
