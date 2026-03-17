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

    delta_a_range = np.linspace(-0.0012, 0.0012, 41)
    rel_f_range   = np.linspace(0.9988, 1.0012, 41)

    ham_type = 'all_ord'
#    phase_arr = phase_residuals_lemniscate_sym(mx_arr, t_arr, a_opt, math.sqrt(2)*fmax_ideal, eta, N_cutoff, ham_type=ham_type)

#    phase_arr     = phase_residuals_lemniscate(mx_arr, t_arr, a_opt, fmax_ideal, eta, N_cutoff)

#    inf = 1 - fidelity_from_phase_residuals(phase_arr)
#    ph_ex = phonon_excitation_prob(phase_arr)

    inf_sym_arr = []#np.zeros((delta_a_range.shape[0], rel_f_range.shape[0]))
    ph_ex_sym_arr = []

    for delta_a in delta_a_range:
        inf_sym_arr.append([])
        ph_ex_arr.append([])
        print(f'delta_a = {delta_a}')
        for rel_f in rel_f_range:
            phase_arr_sym = phase_residuals_lemniscate_sym(mx_arr, t_arr, a_opt + delta_a, math.sqrt(2)*fmax_ideal*rel_f, eta, N_cutoff, ham_type=ham_type)
            inf_sym = 1 - fidelity_from_phase_residuals(phase_arr_sym)
            ph_ex_sym = phonon_excitation_prob(phase_arr_sym)
            print(f'    rel_f = {rel_f}, inf_sym={inf_sym}')
            inf_sym_arr[-1].append(inf_sym)

    inf_sym_arr = np.array(inf_sym_arr)
    print(inf_sym_arr)
    np.savez('../data/inf_2d_dep_lemniscate_sym_20_ions_full_ham_eta_0-03.npz', delta_a_range=delta_a_range, rel_f_range=rel_f_range, inf_arr=inf_sym_arr)

    plot = plt.pcolormesh(delta_a_range, rel_f_range, np.log10(inf_sym_arr.T))
    plt.xlabel(r'$\delta a$')
    plt.ylabel(r'$f/f_{opt}$')
    plt.gca().set_aspect('equal')
    plt.colorbar(plot)
    plt.show()

    #print(inf, ph_ex)
    #print(inf_sym, ph_ex_sym)
