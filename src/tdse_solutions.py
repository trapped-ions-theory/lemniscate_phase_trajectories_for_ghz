import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from qutip import *
from scipy.special import genlaguerre, hyp1f1
from scipy import sparse


def ion_ham_3ord(f_arr, t_arr, eta, m_x, N_cutoff):
    a = destroy(N_cutoff)
    A = a - eta**2/2*a.dag()*a*a
    return QobjEvo([[A,        m_x*np.conj(f_arr)],
                    [A.dag(),  m_x*f_arr]], tlist=t_arr)


def ion_ham_all_ord(f_arr, t_arr, eta, m_x, N_cutoff):
    n_range = np.arange(0, N_cutoff-1)

    #A_diag = math.exp(-eta**2/2)/np.sqrt(n_range+1)*np.array([genlaguerre(n, 1)(eta**2) for n in n_range])
    A_diag = math.exp(-eta**2/2)*np.sqrt(n_range+1)*np.array([hyp1f1(-n, 2, eta**2) for n in n_range])
    #A = -1j/eta*Qobj(np.diag(np.diag((1j*math.sqrt(2)*eta*position(N_cutoff)).expm().data.to_array(), k=1), k=1))
    A = Qobj(sparse.diags(A_diag, offsets=1))
    return QobjEvo([[A,        m_x*np.conj(f_arr)],
                    [A.dag(),  m_x*f_arr]], tlist=t_arr)


def phase_residuals_rect(mx_arr, t_arr, f_max, n_loops, eta, N_cutoff, ham_type='3ord'):
    if ham_type == '3ord':
        ion_ham = ion_ham_3ord
    if ham_type == 'all_ord':
        ion_ham = ion_ham_all_ord

    phase_arr = []
    f_arr = f_max*np.exp(-1j*n_loops*t_arr)

    #N_cutoff = get_coherent_cutoff(alpha_est, 1e-7)
    psi0 = fock(N_cutoff, 0)
    for mx in mx_arr:
        ham = ion_ham(f_arr, t_arr, eta, mx, N_cutoff)
        result = sesolve(ham, psi0, t_arr, options={'store_states'     : False,
                                                    'store_final_state': True})
        phase = result.final_state[0,0]*np.exp(1j*np.pi/2*mx**2)
        phase_arr.append(phase)

    phase_arr = np.array(phase_arr)
    return phase_arr


def phase_residuals_rect_sym(mx_arr, t_arr, f_max, n_loops, eta, N_cutoff, ham_type='3ord'):    
    if ham_type == '3ord':
        ion_ham = ion_ham_3ord
    if ham_type == 'all_ord':
        ion_ham = ion_ham_all_ord

    phase_arr = []
    t_gate = t_arr[-1]
    n_t = t_arr.size
    n_mid = int((n_t - 1)/2)
    t_arr_1 = t_arr[:n_mid+1]
    t_arr_2 = t_arr[n_mid:]

    f_arr_1 =  f_max*np.exp(-2j*n_loops*t_arr_1)
    f_arr_2 = -f_max*np.exp(-2j*n_loops*t_arr_2)

    psi0_1 = fock(N_cutoff, 0)
    for mx in mx_arr:
        ham_1 = ion_ham(f_arr_1, t_arr_1, eta, mx, N_cutoff)
        ham_2 = ion_ham(f_arr_2, t_arr_2, eta, mx, N_cutoff)
        result_1 = sesolve(ham_1, psi0_1, t_arr_1, options={'store_states'     : False, 
                                                    'store_final_state': True})
        psi0_2 = result_1.final_state
        result_2 = sesolve(ham_2, psi0_2, t_arr_2, options={'store_states'     : False, 
                                                    'store_final_state': True})
        phase = result_2.final_state[0,0]*np.exp(1j*np.pi/2*mx**2)
        phase_arr.append(phase)

    phase_arr = np.array(phase_arr)
    return phase_arr


def lemniscate_f(t, a):
    return np.exp(-1j*t) - a*(np.cos(t) - np.cos(2*t))


def phase_residuals_lemniscate(mx_arr, t_arr, a, f_max, eta, N_cutoff, ham_type='3ord'):
    if ham_type == '3ord':
        ion_ham = ion_ham_3ord
    if ham_type == 'all_ord':
        ion_ham = ion_ham_all_ord
    
    phase_arr = []
    f_arr = f_max*lemniscate_f(t_arr, a)
    psi0 = fock(N_cutoff, 0)
    for mx in mx_arr:
        ham = ion_ham(f_arr, t_arr, eta, mx, N_cutoff)
        result = sesolve(ham, psi0, t_arr, options={'store_states' : False, 
                                                'store_final_state': True})
        phase = result.final_state[0,0]*np.exp(1j*np.pi/2*mx**2)
        phase_arr.append(phase)
    
    phase_arr = np.array(phase_arr)
    return phase_arr



#def phase_residuals_lemniscate_sym(mx_arr, t_arr, a, f_max, eta, N_cutoff):
#    phase_arr = []
#    f_arr = f_max*np.where(t_arr < t_arr[-1]/2,
#                           lemniscate_f(2*t_arr, a),
#                          -lemniscate_f(2*t_arr, a))
#    psi0 = fock(N_cutoff, 0)
#    for mx in mx_arr:
#        ham = ion_ham(f_arr, t_arr, eta, mx, N_cutoff)
#        result = sesolve(ham, psi0, t_arr, options={'store_states' : False, 
#                                                'store_final_state': True})
#        phase = result.final_state[0,0]*np.exp(1j*np.pi/2*mx**2)
#        phase_arr.append(phase)
#    
#    phase_arr = np.array(phase_arr)
#    return phase_arr


def phase_residuals_lemniscate_sym(mx_arr, t_arr, a, f_max, eta, N_cutoff, ham_type='3ord'):
    if ham_type == '3ord':
        ion_ham = ion_ham_3ord
    if ham_type == 'all_ord':
        ion_ham = ion_ham_all_ord
    
    phase_arr = []
    t_gate = t_arr[-1]
    n_t = t_arr.size
    n_mid = int((n_t - 1)/2)
    t_arr_1 = t_arr[:n_mid+1]
    t_arr_2 = t_arr[n_mid:]
    
    f_arr_1 = f_max*lemniscate_f(2*t_arr_1, a)
    f_arr_2 = -f_max*(lemniscate_f(2*t_arr_2, a))
    psi0_1 = fock(N_cutoff, 0)
    for mx in mx_arr:
        ham_1 = ion_ham(f_arr_1, t_arr_1, eta, mx, N_cutoff)
        ham_2 = ion_ham(f_arr_2, t_arr_2, eta, mx, N_cutoff)
        result_1 = sesolve(ham_1, psi0_1, t_arr_1, options={'store_states'     : False, 
                                                    'store_final_state': True})
        psi0_2 = result_1.final_state
        result_2 = sesolve(ham_2, psi0_2, t_arr_2, options={'store_states'     : False, 
                                                    'store_final_state': True})
        phase = result_2.final_state[0,0]*np.exp(1j*np.pi/2*mx**2)
        phase_arr.append(phase)
    
    phase_arr = np.array(phase_arr)
    return phase_arr



def fidelity_from_phase_residuals(phase_arr):
    n_ions = phase_arr.size - 1
    overlap = np.sum([math.comb(n_ions, k)/2**n_ions*phase_arr[k] for k in range(n_ions+1)])
    return np.abs(overlap)**2


def phonon_excitation_prob(phase_arr):
    n_ions = phase_arr.size - 1
    return np.sum([math.comb(n_ions, k)/2**n_ions*(1 - abs(phase_arr[k])**2) 
                  for k in range(n_ions+1)])
