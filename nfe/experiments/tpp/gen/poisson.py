"""
Homogeneous Poisson process.
"""
import numpy as np


def sample(_sentinel=None, t_max=None, n_samples=None, lmbd=1.0):
    """Draw samples from the distribution.

    Args:
        t_max: Maximum duration.
        n_samples: Number of points to generate.
        lmbd: Conditional intensity of the point process.

    Returns:
        arrival_times: Arrival times of the points, shape (n_samples).

    """
    if _sentinel is not None:
        raise ValueError("Passing positional arguments is not supported.")

    if t_max is None and n_samples is None:
        raise ValueError(f"Either t_max or n_samples must be specified.")
    elif (t_max is not None) and (n_samples is not None):
        raise ValueError(f"Only one of t_max or n_samples must be specified.")
    elif t_max is not None:
        # Generate more samples than necessary
        n_samples = 1.5 * lmbd * t_max
        inter_times = np.random.exponential(1 / lmbd, size=(n_samples))
        arrival_times = inter_times.cumsum()
        # Only select points before t_max
        arrival_times = arrival_times[arrival_times < t_max]
    elif n_samples is not None:
        inter_times = np.random.exponential(1 / lmbd, size=(n_samples))
        arrival_times = inter_times.cumsum()
    return arrival_times


def nll(arrival_times, lmbd=1.0):
    """Compute negative log-likelihood of a set of points.

    Args:
        arrival_times: Arrival times of the points, shape (n_samples).
        lmbd: Conditional intensity of the point process.

    Returns:
        loss: Negative log-likelihood of the given sequence (scalar).

    """
    t_max = arrival_times.max()
    n_samples = len(arrival_times)
    loss = -np.log(lmbd) + lmbd * t_max / n_samples
    return loss


def poisson(n_samples):
    arrival_times = sample(n_samples=n_samples)
    loss = nll(arrival_times)
    return arrival_times, loss


def intensity(t, arrival_times=None, lmbd=1.0):
    """Compute intensity for each point in the grid.

    Args:
        t: Times for which to compute, shape (n).
        arrival_times: Arrival times of the points, shape (n_samples).
            Not used, only added here for API compatibility with other models.
        lmbd: Conditional intensity of the point process.

    Returns:
        intensity: Intensity values for input times, shape (n).

    """
    return lmbd * np.ones_like(t)
