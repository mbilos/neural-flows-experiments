"""
Hawkes process with exponential kernel.
"""
import numpy as np


def hawkes1(n_samples):
    """Hawkes1 model from Omi et al. 2019."""
    mu = 0.2
    alpha = [0.8, 0.0]
    beta = [1.0, 20.0]
    arrival_times, loglike = _sample_and_nll(n_samples, mu, alpha, beta)
    nll = -loglike.mean()
    return arrival_times, nll


def hawkes2(n_samples):
    """Hawkes2 model from Omi et al. 2019."""
    mu = 0.2
    alpha = [0.4, 0.4]
    beta = [1.0, 20.0]
    arrival_times, loglike = _sample_and_nll(n_samples, mu, alpha, beta)
    nll = -loglike.mean()
    return arrival_times, nll


def _sample_and_nll(n_samples, mu, alpha, beta):
    """Generate samples from Hawkes process & compute NLL.

    Source: https://github.com/omitakahiro/NeuralNetworkPointProcess

    """
    T = []
    LL = []

    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0

    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential()/l
        x = x + step

        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2

        if np.random.rand() < l_next/l: #accept
            T.append(x)
            LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1

            if count == n_samples:
                break

    return [np.array(T), np.array(LL)]
