import numpy as np
from pathlib import Path

from nfe.experiments.tpp.gen.hawkes import hawkes1, hawkes2
from nfe.experiments.tpp.gen.poisson import poisson
from nfe.experiments.tpp.gen.renewal import renewal
from nfe.experiments.tpp.gen.utils import get_inter_times

DATA_DIR = Path(__file__).parents[1] / 'data/tpp'
DATA_DIR.mkdir(parents=True, exist_ok=True)

NUM_SEQUENCES = 1000
NUM_EVENTS = 100

np.random.seed(123)

def get_data(func):
    data, nll = [], []
    for _ in range(NUM_SEQUENCES):
        t, l = func(NUM_EVENTS)
        data.append(get_inter_times(t))
        nll.append(l)
    print(f'{func.__name__} Best loss: {np.mean(nll):.4f} '
          f'(test loss: {np.mean(nll[int(0.8 * NUM_SEQUENCES):]):.4f})')
    return np.array(data), np.array(nll)

def generate():
    filename = DATA_DIR / 'hawkes1.npz'
    if not filename.exists():
        data, nll = get_data(hawkes1)
        np.savez(filename, data=data, nll=nll)

    filename = DATA_DIR / 'hawkes2.npz'
    if not filename.exists():
        data, nll = get_data(hawkes2)
        np.savez(filename, data=data, nll=nll)

    filename = DATA_DIR / 'poisson.npz'
    if not filename.exists():
        data, nll = get_data(poisson)
        np.savez(filename, data=data, nll=nll)

    filename = DATA_DIR / 'renewal.npz'
    if not filename.exists():
        data, nll = get_data(renewal)
        np.savez(filename, data=data, nll=nll)


if __name__ == '__main__':
    generate()


# hawkes1 Best loss: 0.6215 (test loss: 0.6405)
# hawkes2 Best loss: 0.1067 (test loss: 0.1192)
# poisson Best loss: 1.0000 (test loss: 0.9996)
# renewal Best loss: 0.2485 (test loss: 0.2667)
