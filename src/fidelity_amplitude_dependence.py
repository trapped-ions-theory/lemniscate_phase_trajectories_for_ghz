import math
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.optimize import fsolve
from tdse_solutions import *


if __name__ == '__main__':
    t_gate = 2*np.pi
    t_arr = np.linspace(0, t_gate, 401)

    N_cutoff = 300
    eta = 0.03

    n_ions = 20
    mx_arr = np.arange(-n_ions/2, n_ions/2+1)
    n_loops_arr = np.array([1,2,3,6,8])
    #n_loops_arr = np.array([5,8])

    ham_type='all_ord'

    rel_f_range = np.arange(0.99, 1.01, 1e-4)

    inf_arr = []
    ph_ex_arr = []
    for n_loops in n_loops_arr:
        print(n_loops)
        inf_arr.append([])
        ph_ex_arr.append([])
        #f_range = math.sqrt(n_loops)/2*rel_f_range
        for rel_f in rel_f_range:
            #f = math.sqrt(n_loops)/2*rel_f
            #phase_residuals = phase_residuals_rect(mx_arr, t_arr, f, n_loops, eta, N_cutoff, ham_type=ham_type)
            f = math.sqrt(2*n_loops)/2*rel_f
            phase_residuals = phase_residuals_rect_sym(mx_arr, t_arr, f, n_loops, eta, N_cutoff, ham_type=ham_type)
            inf_arr[-1].append(1 - fidelity_from_phase_residuals(phase_residuals))
            ph_ex_arr[-1].append(phonon_excitation_prob(phase_residuals))
            print(f'rel_f = {rel_f:.5f}, inf = {inf_arr[-1][-1]:.5e}')
        print()

    inf_arr   = np.array(inf_arr)
    ph_ex_arr = np.array(ph_ex_arr)
    np.savez('../data/inf_amplitude_dep_rect_sym_pulse_20_ions_full_ham_eta_0-03.npz', rel_f_range=rel_f_range,
             inf_arr=inf_arr, n_loops_arr=n_loops_arr)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(rel_f_range, inf_arr.T)
    ax2.plot(rel_f_range, ph_ex_arr.T)

    ax1.set_yscale('log')
    ax2.set_yscale('log')

    plt.show()
