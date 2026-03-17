import math
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.optimize import fsolve
from tdse_solutions import *


def get_phase_res_arr(n_loops_arr, rel_f_range, mx_arr, t_arr, eta, N_cutoff, pulse_type, ham_type, print_inf=False):
    phase_res_arr = []

    for n_loops in n_loops_arr:
        print(n_loops)
        phase_res_arr.append([])
        for rel_f in rel_f_range:
            if pulse_type == 'rect':
                f = math.sqrt(n_loops)/2*rel_f
                phase_residuals = phase_residuals_rect(mx_arr, t_arr, f, n_loops, eta, N_cutoff, ham_type=ham_type)
            elif pulse_type == 'rect_sym':
                f = math.sqrt(2*n_loops)/2*rel_f
                phase_residuals = phase_residuals_rect_sym(mx_arr, t_arr, f, n_loops, eta, N_cutoff, ham_type=ham_type)
            if print_inf == True:
                print(1 - fidelity_from_phase_residuals(phase_residuals))

            phase_res_arr[-1].append(phase_residuals)
            print(f'n_loops = {n_loops} rel_f = {rel_f:.5f}')
        print()

    phase_res_arr = np.array(phase_res_arr)
    return phase_res_arr


if __name__ == '__main__':
    t_gate = 2*np.pi
    t_arr = np.linspace(0, t_gate, 401)

    N_cutoff = 300
    eta = 0.03

    n_ions = 20
    mx_arr = np.arange(-n_ions/2, n_ions/2+1)
    n_loops_arr = np.array([1,2,3,6,8])

    ham_type='all_ord'
    rel_f_range = np.arange(0.99, 1.01, 1e-4)

#    print('rect pulse')
#    phase_res_arr_rect = get_phase_res_arr(phase_residuals_rect, n_loops_arr, rel_f_range, mx_arr, t_arr, eta, N_cutoff, ham_type)
#    np.savez('../data/phase_res_amplitude_dep_rect_pulse_20_ions_full_ham_eta_0-03.npz', rel_f_range=rel_f_range,
#             phase_res_arr=phase_res_arr_rect, n_loops_arr=n_loops_arr)

    print('rect sym pulse')
    phase_res_arr_rect_sym = get_phase_res_arr(n_loops_arr, rel_f_range, mx_arr, t_arr,
                                               eta, N_cutoff, 'rect_sym', ham_type, print_inf=True)
    np.savez(f'../data/phase_res_amplitude_dep_rect_sym_pulse_20_ions_full_ham_eta_{str(eta).replace('.', '-')}.npz', rel_f_range=rel_f_range,
             phase_res_arr=phase_res_arr_rect_sym, n_loops_arr=n_loops_arr)
