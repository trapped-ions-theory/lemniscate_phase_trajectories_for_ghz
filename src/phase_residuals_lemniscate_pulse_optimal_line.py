from fidelity_amplitude_dependence import *
import matplotlib.pyplot as plt


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

    delta_a_range = np.linspace(-0.001, 0.005, 61)
    delta_f_range = 0.0006173800113121406 + 1.7634105025990197*delta_a_range

    print('delta f limits:', delta_f_range[0], delta_f_range[-1])

    ham_type = 'all_ord'
    fock0_proj_arr = []

    for delta_a, delta_f in zip(delta_a_range, delta_f_range):
        rel_f = 1 + delta_f
        print(f'delta_a = {delta_a}')

#        fock0_projections = phase_residuals_lemniscate(mx_arr, t_arr, a_opt + delta_a, fmax_ideal*rel_f, eta, N_cutoff, ham_type=ham_type)
        fock0_projections = phase_residuals_lemniscate_sym(mx_arr, t_arr, a_opt + delta_a, math.sqrt(2)*fmax_ideal*rel_f, eta, N_cutoff, ham_type=ham_type)
        fock0_proj_arr.append(fock0_projections)
        inf_sym = 1 - fidelity_from_phase_residuals(fock0_projections)
        print(f'    rel_f - 1 = {rel_f-1:.2e}, inf_sym={inf_sym:2e}')

    fock0_proj_arr = np.array(fock0_proj_arr)
#    np.savez(f'../data/phase_res_2d_dep_lemniscate_20_ions_full_ham_eta_{str(eta).replace('.', '-')}.npz', delta_a_range=delta_a_range, rel_f_range=rel_f_range, phase_res_arr=phase_res_arr)
    #np.savez('../data/phase_res_2d_dep_lemniscate_sym_20_ions_full_ham_eta_0-03.npz', delta_a_range=delta_a_range, rel_f_range=rel_f_range, phase_res_arr=phase_res_arr)
