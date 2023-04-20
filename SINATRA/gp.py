#!/bin/python3

import sys
import numpy as np
from scipy.stats.distributions import chi2
from scipy.stats import norm
#from sinatra_pro.RATE import *
import RATE



def CovarianceMatrix(x, bandwidth=0.01):
    """
    Computes the covariance matrix given by the Gaussian kernel.

    `X` is the design matrix where columns are observations.

    `bandwidth` is the free parameter for the Gaussian kernel.
    """
    bandwidth = 1. / (2 * bandwidth ** 2)
    n = x.shape[1]
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        K[i, i] = 1
        for j in range(i + 1, n):
            K[i, j] = np.exp(-np.mean((x[:, i] - x[:, j]) ** 2) * bandwidth)
            K[j, i] = K[i, j]
    return K

def probit_log_likelihood(latent_variables, y):
    """Probit link function"""
    return np.sum(np.log(norm.cdf(latent_variables * y)))

def logistic_log_likelihood(latent_variables, y):
    """Logistic link function"""
    return (-np.sum(np.log(1.0 + np.exp(latent_variables * y))))

def linear_log_likelihood(latent_variables, y):
    Y_mean = y.mean()
    Y_std = y.std()
    Z = (latent_variables - Y_mean) / Y_std
    lin_lik = norm.sf(abs(Z))
    return np.sum(lin_lik)

def Elliptical_Slice_Sampling(K, y, func, n_mcmc=100000, burn_in=1000, seed=None, verbose=False):
    """
    Elliptical slice sampling algorithm adopted from FastGP::ess from FastGP R package.
    The function returns the desired number of MCMC samples.

    `K` is the covariance matrix of the Gaussian Process model.

    `y` is the list of binary class labels for each data points, 0 or 1.

    `n_mcmc` is the number of desired mcmc samples to be returned.

    `burn_in` is the number of burn in steps before MCMC sampling.

    By default, the likelihood function uses probit link. If `probit` is set to False, the function uses logistic link instead.

    If `seed` is provided, it will be set as the seed for the random number generator (for testing purpose).

    If `verbose` is set to True, the program prints progress on command prompt.
    """

    if verbose:
        print("Running elliptical slice sampling...")
    if func=="probit":
        log_lik = probit_log_likelihood
    if func=="logistic":
        log_lik = logistic_log_likelihood
    if func=="linear":
        log_lik = linear_log_likelihood
    n = K.shape[0]
    N = y.size
    print("n", n)
    print("N", N)
    if isinstance(seed, int):
        np.random.seed(seed)
    mcmc_samples = np.zeros(shape= (burn_in + n_mcmc, N), dtype=float)
    norm_samples = np.random.multivariate_normal(mean=np.zeros(n), cov=K, size=burn_in + n_mcmc)
    unif_samples = np.random.uniform(low=0, high=1, size=burn_in + n_mcmc)
    theta = np.random.uniform(low=0, high=2 * np.pi, size=burn_in + n_mcmc)
    theta_min = theta - 2 * np.pi
    theta_max = theta + 2 * np.pi
    print("mcmc", mcmc_samples.shape)
    print("norm", norm_samples.shape)
    for i in range(1, burn_in + n_mcmc):
        if verbose:
            if i < burn_in:
                sys.stdout.write('Burning in...\r')
            #else:
                #sys.stdout.write('Elliptical slice sampling Step %d...\r' % (i - burn_in + 1))
            sys.stdout.flush()
        f = mcmc_samples[i - 1, :]
        #ADDED LINES BELOW
        llh_thresh = log_lik(f, y) + np.log(unif_samples[i])
        f_star = f * np.cos(theta[i]) + norm_samples[i, :] * np.sin(theta[i])
        while (log_lik(f_star, y) < llh_thresh):
            if theta[i] < 0:
                theta_min[i] = theta[i]
            else:
                theta_max[i] = theta[i]
            theta[i] = np.random.uniform(low=theta_min[i], high=theta_max[i], size=1)
            f_star = f * np.cos(theta[i]) + norm_samples[i, :] * np.sin(theta[i])
        mcmc_samples[i, :] = f_star
    if verbose:
        sys.stdout.write('\n')
    return mcmc_samples[burn_in:, :]

def DGP_samples():
    ms = np.array(np.loadtxt("/users/bsun14/DGP_out/trial_1/m.txt"))
    vs = np.array(np.loadtxt("/users/bsun14/DGP_out/trial_1/v.txt"))
    samples = []
    sample = []
    for m, v in zip(ms, vs):
        sample = []
        sample = (np.random.normal(m, np.sqrt(v), int(100000)))
        samples.append(sample)
    samples = np.array(samples).T
    print(samples.shape)
    return samples

def calc_rate(X, y, func, bandwidth=0.01, sampling_method='ESS', n_mcmc=100000, burn_in=1000, probit=True, seed=None,
              prop_var=1, low_rank=False, parallel=False, n_core=-1, verbose=False):
    """
    Calculate RelATive cEntrality (RATE) centrality measures from data.

    `X` is the design matrix where columns are observations.

    `y` is the list of the class labels for each data points, 0 or 1.

    `bandwidth` is the free parameter for the Gaussian kernel.

    `sampling_method` is the sampling method used for sampling parameters, currently only `ESS` (elliptical slice sampling) is available.

    `n_mcmc` is the number of desired MCMC samples to be returned.

    `burn_in` is the number of burn in steps before MCMC sampling.

    By default, the likelihood function uses probit link. If `probit` is set to False, the function uses logistic link instead.

    If `seed` is provided, it will be set as the seed for the random number generator (for testing purpose).

    'prop_var' is the desired proportion of variance in RATE calculation that the user wants to explain when applying singular value decomposition (SVD) to the design matrix X (this is preset to 1);

    'low_rank' is a boolean variable detailing if the function will use low rank matrix approximations to compute the RATE values --- note that this highly recommended in the case that the number of covariates (e.g. SNPs, genetic markers) is large;

    If `parallel` is set to True, the program runs on multiple cores for RATE calculations,
    then `n_core` will be the number of cores used (the program uses all detected cores if `n_core` is not provided).

    If `verbose` is set to True, the program prints progress on command prompt.

    """
    n = X.shape[0]
    f = np.zeros(n)
    if verbose:
        sys.stdout.write('Calculating Covariance Matrix...\n')
    Kn = CovarianceMatrix(X.T, bandwidth)
    samples = Elliptical_Slice_Sampling(Kn, y, func, n_mcmc=n_mcmc, burn_in=burn_in, seed=seed,
                                        verbose=verbose)

    #samples = DGP_samples()
    print("Elliptical Slice Sampling done")

    #np.savetxt('/users/bsun14/sp/trial_10/gp_X.txt', X)
    #np.savetxt('/users/bsun14/sp/trial_10/gp_samples.txt', samples)
    kld, rates, delta, eff_samp_size = RATE.calc_RATE(X=X, f_draws=samples, prop_var=prop_var, low_rank=low_rank,
                                            parallel=parallel, n_core=n_core, verbose=verbose)
    # TESTING WHAT THE ERROR IS HERE
    #kld, rates, delta, eff_samp_size = RATE_tch.calc_RATE(X=X, f_draws=samples, prop_var=prop_var, low_rank=low_rank,
                                                #parallel=parallel, n_core=n_core, verbose=verbose)

    return kld, rates, delta, eff_samp_size

def calc_rate_test(gp_X, gp_samples, prop_var=1, low_rank=False, parallel=False, n_core=-1, verbose=False):
    X = np.loadtxt(gp_X)
    samples = np.loadtxt(gp_samples)
    kld, rates, delta, eff_samp_size = RATE_tch.calc_RATE(X=X, f_draws=samples, prop_var=prop_var, low_rank=low_rank,
                                                          parallel=parallel, n_core=n_core, verbose=verbose)

    return kld, rates, delta, eff_samp_size


