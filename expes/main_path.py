"""Compute ElasticNet path with SSNAL algorithm."""


import numpy as np
from numpy import linalg as LA
from scipy.linalg import eigh as largest_eigh

from ssnal import ssnal_elastic_path


if __name__ == '__main__':
    seed = 54
    np.random.seed(seed)

    # --------------------------- #
    #  set simulation parameters  #
    # --------------------------- #

    # observations, features, sparsity
    m, n, non_zeros = 500, 10_000, 10

    # x true
    xstar = 5

    # signal to noise ratio
    snr = 5

    # --------------------- #
    #  set ssnal_parameter  #
    # --------------------- ##

    # set a vector with lam1_max percentages to consider
    c_lam_vec = np.geomspace(1, 0.1, num=100)

    # alpha
    alpha = 0.6

    # max selected features
    max_selected = 100

    # decide if cv is performed
    cv = True

    # number of folds for cv
    n_folds = 10

    # to choose between cg and exact method
    use_cg = True
    r_exact = 2000  # number of selected features such that we start using the exact method

    # to control sigma
    sgm = 0.005
    sgm_increase = 5
    sgm_change = 1

    # to select step size in line search
    mu = 0.2
    step_reduce = 0.5

    # max iterations and tolerance
    maxiter_ssn = 40
    maxiter_ssnal = 100
    tol_ssn = 1e-6
    tol_ssnal = 1e-6

    # decide if a plot is displayed
    plot = True

    # decide level of printing
    print_lev = 2

    # ---------------------------------- #
    #  computing matrices and variables  #
    # ---------------------------------- #

    # create design matrix A
    print('')
    print('  * generating A')
    A = np.random.normal(0, 1, (m, n))
    print('')

    # compute true coefficients
    x_true = np.zeros((n,))
    x_true[0:non_zeros] = xstar

    # compute err variance and error
    err_sd = np.sqrt(np.var(np.dot(A, x_true)) / snr)
    err = err_sd * np.random.normal(0, 1, (m, ))

    # compute the response
    b = np.dot(A, x_true).reshape(m, ) + err
    b -= np.mean(b)

    # -------------------- #
    #  ssnal elastic path  #
    # -------------------- #

    out_path = ssnal_elastic_path(
        A=A, b=b, c_lam_vec=c_lam_vec, alpha=alpha, max_selected=max_selected,
        cv=cv, n_folds=n_folds, x0=None, y0=None, z0=None, Aty0=None,
        sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
        step_reduce=step_reduce, mu=mu, tol_ssn=tol_ssn, tol_ssnal=tol_ssnal,
        maxiter_ssn=maxiter_ssn, maxiter_ssnal=maxiter_ssnal,
        use_cg=use_cg, r_exact=r_exact, plot=plot, print_lev=print_lev)

    # ------------------------------------ #
    #  if I want to test different alphas  #
    # ------------------------------------ #

    # # values of alphas
    # alpha_vec = np.array([0.9, 0.6, 0.3])
    #
    # # create list to plot
    # r_lm_list, ebic_list, gcv_list, cv_vec_list = list(), list(), list(), list()
    #
    # # loop for alphas values
    # for i in range(alpha_vec.shape[0]):
    #     out_path_alpha = ssnal_elastic_path(A=A, b=b,
    #                                         c_lam_vec=c_lam_vec, alpha=alpha_vec[i],
    #                                         max_selected=max_selected,
    #                                         cv=cv, n_folds=n_folds,
    #                                         x0=None, y0=None, z0=None, Aty0=None,
    #                                         sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
    #                                         step_reduce=step_reduce, mu=mu,
    #                                         tol_ssn=tol_ssn, tol_ssnal=tol_ssnal,
    #                                         maxiter_ssn=maxiter_ssn, maxiter_ssnal=maxiter_ssnal,
    #                                         use_cg=use_cg, r_exact=r_exact,
    #                                         plot=False, print_lev=print_lev)
    #
    #     r_lm_list.append(out_path_alpha[5])
    #     ebic_list.append(out_path_alpha[1])
    #     gcv_list.append(out_path_alpha[2])
    #     cv_vec_list.append(out_path_alpha[3])
    #
    # import auxiliary_functions as AF
    # AF.plot_cv_ssnal_elstic(r_lm_list, ebic_list, gcv_list, cv_vec_list, alpha_vec, c_lam_vec)
