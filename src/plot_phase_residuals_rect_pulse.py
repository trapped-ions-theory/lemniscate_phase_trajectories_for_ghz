import math
import numpy as np
import matplotlib.pyplot as plt

from tdse_solutions import ion_ham, phase_residuals_rect

if __name__ == '__main__':
    t_gate = 2*np.pi
    t_arr = np.linspace(0, t_gate, 4001)
    
    N_cutoff = 600
    eta = 0.03
    
    n_ions = 20
    mx_arr = np.arange(-n_ions/2, n_ions/2+1)
    f_max = 1/2
    n_loops = 1
    phase_arr_rect_1 = phase_residuals_rect(mx_arr, t_arr, f_max, n_loops, eta, N_cutoff)
    phases_theory_rect_1 = np.exp(3j*np.pi/8*eta**2*mx_arr**4)

    plt.scatter(mx_arr, phase_arr_rect_1.real)
    plt.scatter(mx_arr, phase_arr_rect_1.imag)
    plt.plot(mx_arr, phases_theory_rect_1.real)
    plt.plot(mx_arr, phases_theory_rect_1.imag)
    plt.show()