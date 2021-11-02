"""
Renewal process. Conditional probability is fixed to f*(t) = lognormal(t).
"""
import numpy as np
from scipy.stats import lognorm

from nfe.experiments.tpp.gen.utils import get_inter_times

def sample(n_samples, std=6):
    """Draw samples from the distribution.

    Args:
        n_samples: Number of samples to generate.
        std: Standart deviation of f*(t).

    """
    s = np.sqrt(np.log(std**2 + 1))
    mu = -0.5 * s * s
    inter_times = lognorm.rvs(s=s, scale=np.exp(mu),size=n_samples)
    arrival_times = inter_times.cumsum()
    return arrival_times


def nll(arrival_times, std=6):
    """Negative log-likelihood of a renewal process.

    Conditional density f*(t) is lognormal with given std.

    """
    s = np.sqrt(np.log(std**2 + 1))
    mu = -0.5 * s * s
    inter_times = get_inter_times(arrival_times)
    log_probs = lognorm.logpdf(inter_times, s=s, scale=np.exp(mu))
    return -np.mean(log_probs)


def renewal(n_samples):
    arrival_times = sample(n_samples)
    loss = nll(arrival_times)
    return arrival_times, loss


def intensity(t, arrival_times, std=6):
    """Compute intensity for each point in the grid.

    Args:
        t: Times for which to compute, shape (n).
        arrival_times: Arrival times of the points, shape (n_samples).
        lmbd: Conditional intensity of the point process.

    Returns:
        intensity: Intensity values for input times, shape (n).
    """
    # Compute time since last event for each grid point
    delta = t.reshape(-1, 1) - arrival_times
    delta[delta < 0] = np.inf
    time_since_last = delta.min(1)
    time_since_last[time_since_last == np.inf] = 0.0
    # Compute PDF & CDF at each grid point
    s = np.sqrt(np.log(std**2 + 1))
    mu = -0.5 * s * s
    pdf = lognorm.pdf(time_since_last, s=s, scale=np.exp(mu))
    cdf = lognorm.cdf(time_since_last, s=s, scale=np.exp(mu))
    return pdf / (1 - cdf)
