import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 20})
    np.set_printoptions(linewidth=300, precision=10)


    datafile = np.load('../data/inf_amplitude_dep_rect_pulse_20_ions_eta_0-03.npz')
