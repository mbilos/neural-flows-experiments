import numpy as np

def get_inter_times(arrival_times):
    """Convert arrival times to interevent times."""
    return arrival_times - np.concatenate([[0], arrival_times[:-1]])
