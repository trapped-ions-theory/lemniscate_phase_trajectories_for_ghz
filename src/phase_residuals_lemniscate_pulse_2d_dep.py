from fidelity_amplitude_dependence import *
import matplotlib.pyplot as plt
#from multiprocessing import Pool


if __name__ == '__main__':
    t_gate = 2*np.pi
    t_arr = np.linspace(0, t_gate, 401)

    N_cutoff = 300
    eta = 0.03

    n_ions = 20
    mx_arr = np.arange(-n_ions/2, n_ions/2+1)

    inf_arr = []
    ph_ex_arr = []


    a_opt = 0.7274788716591838
    fmax_ideal = 0.95778915

    delta_a_range = np.arange(-0.001, 0.006, 1e-4)
    rel_f_range   = 1 + np.arange(-0.002, 0.012, 2e-4)
    #rel_f_range   = 1 + np.linspace(0.00, 0.001, 21)

    ham_type = 'all_ord'

    fock0_proj_arr = []


    for delta_a in delta_a_range:
        fock0_proj_arr.append([])
        print(f'delta_a = {delta_a}')
        for rel_f in rel_f_range:
            #fock0_projections = phase_residuals_lemniscate(mx_arr, t_arr, a_opt + delta_a, fmax_ideal*rel_f, eta, N_cutoff, ham_type=ham_type)
            fock0_projections = phase_residuals_lemniscate_sym(mx_arr, t_arr, a_opt + delta_a, math.sqrt(2)*fmax_ideal*rel_f, eta, N_cutoff, ham_type=ham_type)
            fock0_proj_arr[-1].append(fock0_projections)
            inf_sym = 1 - fidelity_from_phase_residuals(fock0_projections)
            print(f'    rel_f - 1 = {rel_f-1:.2e}, inf_sym={inf_sym:2e}')

    fock0_proj_arr = np.array(fock0_proj_arr)
#    np.savez(f'../data/phase_res_2d_dep_lemniscate_20_ions_full_ham_eta_{str(eta).replace('.', '-')}.npz', delta_a_range=delta_a_range, rel_f_range=rel_f_range, phase_res_arr=fock0_proj_arr)
    np.savez(f'../data/phase_res_2d_dep_lemniscate_sym_20_ions_full_ham_eta_{str(eta).replace('.', '-')}.npz',
            delta_a_range=delta_a_range, rel_f_range=rel_f_range, phase_res_arr=fock0_proj_arr)
