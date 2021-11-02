import numpy as np
import scipy.signal
import scipy.integrate
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / 'data/synth'
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(123)

NUM_SEQUENCES = 1000
NUM_POINTS = 100
MAX_TIME = 10
EXTRAPOLATION_TIME = 20

def get_inital_value(extrap_space):
    if extrap_space:
        return np.random.uniform(-4, -2, (1,)) if np.random.rand() > 0.5 else np.random.uniform(2, 4, (1,))
    else:
        return np.random.uniform(-2, 2, (1,))

def get_inital_value2d(extrap_space):
    if extrap_space:
        return np.random.uniform(1, 2, (2,))
    else:
        return np.random.uniform(0, 1, (2,))

def get_data(func, time_min, time_max, extrap_space=False, name=None):
    initial_values = []
    times = []
    sequences = []

    for _ in range(NUM_SEQUENCES):
        t = np.sort(np.random.uniform(time_min, time_max, NUM_POINTS))
        y0, y = func(t, extrap_space)
        times.append(t)
        initial_values.append(y0)
        sequences.append(y)

    initial_values, times, sequences = np.array(initial_values), np.array(times), np.array(sequences)
    if name is None:
        return initial_values, times, sequences
    else:
        np.savez(DATA_DIR / f'{name}.npz', init=initial_values, seq=sequences, time=times)

def generate():
    # SINE
    def sine_func(t, extrap_space=False):
        y = get_inital_value(extrap_space)
        return y, np.sin(t[:,None]) + y
    if not (DATA_DIR / 'sine.npz').exists():
        get_data(sine_func, 0, MAX_TIME, name='sine')
        get_data(sine_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='sine_extrap_time')
        get_data(sine_func, 0, MAX_TIME, extrap_space=True, name='sine_extrap_space')

    # SQUARE
    def square_func(t, extrap_space=False):
        y = get_inital_value(extrap_space)
        return y, np.sign(np.sin(t[:,None])) + y
    if not (DATA_DIR / 'square.npz').exists():
        get_data(square_func, 0, MAX_TIME, name='square')
        get_data(square_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='square_extrap_time')
        get_data(square_func, 0, MAX_TIME, extrap_space=True, name='square_extrap_space')

    # SAWTOOTH
    def sawtooth_func(t, extrap_space=False):
        y = get_inital_value(extrap_space)
        return y, scipy.signal.sawtooth(t[:,None]) + y
    if not (DATA_DIR / 'sawtooth.npz').exists():
        get_data(sawtooth_func, 0, MAX_TIME, name='sawtooth')
        get_data(sawtooth_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='sawtooth_extrap_time')
        get_data(sawtooth_func, 0, MAX_TIME, extrap_space=True, name='sawtooth_extrap_space')

    # TRIANGLE
    def triangle_func(t, extrap_space=False):
        y = get_inital_value(extrap_space)
        return y, np.abs(scipy.signal.sawtooth(t[:,None])) + y
    if not (DATA_DIR / 'triangle.npz').exists():
        get_data(triangle_func, 0, MAX_TIME, name='triangle')
        get_data(triangle_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='triangle_extrap_time')
        get_data(triangle_func, 0, MAX_TIME, extrap_space=True, name='triangle_extrap_space')


    # SINK
    def sink_func(t, extrap_space=False):
        y = get_inital_value2d(extrap_space)
        ode = lambda y, t: np.array([[-4, 10], [-3, 2]]) @ y
        return y, scipy.integrate.odeint(ode, y, t)
    if not (DATA_DIR / 'sink.npz').exists():
        get_data(sink_func, 0, MAX_TIME, name='sink')
        get_data(sink_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='sink_extrap_time')
        get_data(sink_func, 0, MAX_TIME, extrap_space=True, name='sink_extrap_space')

    # ELLIPSE (Lotka-Volterra)
    def ellipse_func(t, extrap_space=False):
        y = get_inital_value2d(extrap_space)
        ode = lambda y, t: np.array([2/3 * y[0] - 2/3 * y[0] * y[1], y[0] * y[1] - y[1]])
        return y, scipy.integrate.odeint(ode, y, t) - 1
    if not (DATA_DIR / 'ellipse.npz').exists():
        get_data(ellipse_func, 0, MAX_TIME, name='ellipse')
        get_data(ellipse_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='ellipse_extrap_time')
        get_data(ellipse_func, 0, MAX_TIME, extrap_space=True, name='ellipse_extrap_space')


if __name__ == '__main__':
    generate()
