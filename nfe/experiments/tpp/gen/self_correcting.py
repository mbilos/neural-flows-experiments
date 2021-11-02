"""
Self-correcting process.
Conditional intensity is lambda*(t) = exp(m*t - sum_{ti < t} alpha).
"""
import numpy as np
from nfe.experiments.tpp.gen.utils import get_inter_times

def sample(n_samples, mu=1.0, alpha=1.0):
    """Draw samples from the distribution.

    Args:
        n_samples: Number of points to generate.
        mu: Base rate.
        alpha: Decay from each event.

    Returns:
        arrival_times: Arrival times of the points, shape (n_samples).

    """
    x = 0
    inter_times = []

    for i in range(n_samples):
        e = np.random.exponential()
        tau = np.log(e * mu / np.exp(x) + 1 ) / mu
        # Equivalent to e = (np.exp(mu * tau) - 1) * np.exp(x) / mu
        inter_times.append(tau)
        x = x + mu * tau - alpha

    inter_times = np.array(inter_times)
    return inter_times.cumsum()


def nll(arrival_times, mu=1.0, alpha=1.0):
    """Compute negative log-likelihood of a set of points.

    Args:
        arrival_times: Arrival times of the points, shape (n_samples).
        lmbd: Conditional intensity of the point process.

    Returns:
        loss: Negative log-likelihood of the given sequence (scalar).

    """
    inter_times = get_inter_times(arrival_times)
    # Loss for the log-intensity part (-sum_i log(lambda(t_i)))
    counts = np.arange(len(arrival_times))
    log_l = alpha * counts - mu * arrival_times

    # Loss for the integral part
    t_end = arrival_times
    t_start = np.concatenate([[0], arrival_times[:-1]])
    int_l = (np.exp(mu * t_end - alpha * counts) - np.exp(mu * t_start - alpha * counts)) / mu
    return np.mean(log_l + int_l)


def intensity(t, arrival_times, mu=1.0, alpha=1.0):
    """Compute intensity for each point in the grid.

    Args:
        t: Times for which to compute, shape (n).
        arrival_times: Arrival times of the points, shape (n_samples).

    Returns:
        intensity: Intensity values for input times, shape (n).

    """
    t_max = arrival_times.max()
    n_events = ((t.reshape(-1, 1) - arrival_times) > 0).sum(1)
    return np.exp(mu * t - n_events * alpha)
