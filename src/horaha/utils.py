import pathlib
import pickle

import numpy as np
import scipy.stats

from horaha import PROJECT_ROOT


def load_monk_data(root_fp=None):
    if root_fp is None:
        root_fp = PROJECT_ROOT
    data_fp = pathlib.Path(root_fp).joinpath('data', 'sampson.txt')
    data = np.loadtxt(str(data_fp))
    return data


def load_eecs_data(root_fp=None):
    if root_fp is None:
        root_fp = PROJECT_ROOT
    data_fp = pathlib.Path(root_fp).joinpath('data', 'Y.pkl')
    with data_fp.open('rb') as f:
        data = pickle.load(f)

    # index out all-zero entries
    inds = np.where(np.any(data > 0, axis=1))[0]
    data = data[inds, :][:, inds]

    return data, inds


def get_sample_parameters(n, k, seed=6882):
    np.random.seed(seed)
    μ = 0
    σ = 100
    pα = scipy.stats.norm(loc=μ, scale=σ)
    α = pα.rvs(size=1)
    Z = pα.rvs(size=(k, n))
    return α, Z


def compute_posterior_means(Z):
    return np.mean(Z, axis=0)
