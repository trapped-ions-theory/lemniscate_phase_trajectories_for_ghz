import numpy as np
import math
import pandas as pd
from scipy.interpolate import PPoly
from scipy.integrate import quad
from qutip import *

#from float_formatters import smart_float_format


def sx4_corr_coef(n_ions):
    Sx = jmat(n_ions/2, 'x')
    psi0 = fock(n_ions+1, 0)
    corr_coef = ((psi0.dag()*Sx**6*psi0 -  psi0.dag()*Sx**4*psi0*psi0.dag()*Sx**2*psi0) / 
                 (psi0.dag()*Sx**4*psi0 - (psi0.dag()*Sx**2*psi0)**2)).real
    return corr_coef


def sx4_residual_inf_pert_theory(eta, n_ions):
    Sx = jmat(n_ions/2, 'x')
    psi0 = fock(n_ions+1, 0)
    H = (psi0.dag()*Sx**8*psi0 - (psi0.dag()*Sx**4*psi0)**2
     - ((psi0.dag()*Sx**6*psi0 -  psi0.dag()*Sx**4*psi0*psi0.dag()*Sx**2*psi0)**2 /
                 (psi0.dag()*Sx**4*psi0 - (psi0.dag()*Sx**2*psi0)**2))).real
    return (0.921*eta**2)**2*H


def sx4_residual_inf_matrix(eta, n_ions):
    Sx = jmat(n_ions/2, 'x')
    theta_4 = 0.921*eta**2
    theta_2 = -sx4_corr_coef(n_ions)*theta_4
    U = (-1j*(theta_2*Sx**2 + theta_4*Sx**4)).expm()
    psi0 = fock(n_ions + 1, 0)
    inf = 1 - np.abs(psi0.dag()*U*psi0)**2
    return inf


def rect_s3_mat(T, epsilon):
    def smoothstep_mat(t_mid, tau):
        t_bp = np.array([0, tau, t_mid + tau, t_mid + 2*tau], dtype=np.float64)
        smoothstep_mat = np.array([[-2/tau**3, 3/tau**2, 0, 0],
                              [0, 0, 0, 1],
                              [2/tau**3, -3/tau**2, 0, 1]], dtype=np.float64).T
        return t_bp, smoothstep_mat
    t_mid = T*(1-2*epsilon)
    tau = epsilon*T
    t_bp, Omega_mat = smoothstep_mat(t_mid, tau)
    return t_bp, Omega_mat


def cmplx_quad(f, a, b):
    return quad(lambda x: f(x).real, a, b)[0] + 1j*quad(lambda x: f(x).imag, a, b)[0]


def phonon_excitation_prob_coef():
    t_gate = 1
    epsilon = 1/3
    t_bp, s_mat = rect_s3_mat(t_gate, epsilon)
    delta = 2*np.pi/t_gate*3/2
    Omega_mat = np.pi/t_gate*1.5258356*s_mat
    Omega = PPoly.construct_fast(Omega_mat, t_bp)

    def alpha_slow(Omega, delta, t):
        return cmplx_quad(lambda t1: -1j/2*Omega(t1)*np.exp(-1j*delta*t1), 0, t)

    def integrand(t):
        return -2j*Omega(t)*(           alpha_slow(Omega, delta, t)**2 *np.exp( 1j*delta*t)
                             + 2*np.abs(alpha_slow(Omega, delta, t)**2)*np.exp(-1j*delta*t))

    return np.abs(cmplx_quad(integrand, 0, t_gate))**2


def phonon_excitation(eta, n_ions, ph_ex_coef):
    Sx = jmat(n_ions/2, 'x')
    psi0 = fock(n_ions+1, 0)
    return (psi0.dag()*Sx**6*psi0*eta**4*ph_ex_coef).real


if __name__ == '__main__':
    eta = 0.05
    n_ions_arr = np.array([2,4,6,8,10,12,16,20,26,30])

    inf_t   = []
    inf_mat = []
    p_ex    = []

    ph_ex_coef = 1.655#phonon_excitation_prob_coef()
    print(ph_ex_coef)

    for n_ions in n_ions_arr:
        inf_t.append(  sx4_residual_inf_pert_theory(eta, n_ions))
        inf_mat.append(sx4_residual_inf_matrix(eta, n_ions))
        p_ex.append(phonon_excitation(eta, n_ions, ph_ex_coef))

    df = pd.DataFrame({'n_ions' : n_ions_arr, 'inf_t' :  inf_t, 'inf_mat' : inf_mat, 'p_ex' : p_ex})
    df.set_index('n_ions', inplace=True)

    print(df.to_latex())
    #print(df.to_latex(float_format=lambda t: smart_float_format(t)))
