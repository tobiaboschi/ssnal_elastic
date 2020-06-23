
# ---------------------------------------------- #
#                                                #
#    Run ssnal_elastic_core on synthetic data    #
#                                                #
# ---------------------------------------------- #

import numpy as np
from numpy import linalg as LA
from ssnal_elastic_core import ssnal_elastic_core
import time
from scipy.linalg import eigh as largest_eigh


if __name__ == '__main__':

    # seed = np.random.randint(0, 1e5)
    seed = 54
    np.random.seed(seed)

    # --------------------------- #
    #  set simulation parameters  #
    # --------------------------- #

    # observations
    m = 500

    # features
    n = 10000

    # sparsity
    non_zeros = 100

    # x true
    xstar = 5

    # signal to noise ratio
    snr = 5

    # --------------------- #
    #  set ssnal_parameter  #
    # --------------------- #

    # set c_lam
    c_lam = 0.7

    # set alpha
    alpha = 0.9

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

    # decide level of printing
    print_lev = 3

    # ---------------------------------- #
    #  computing matrices and variables  #
    # ---------------------------------- #

    # create design matrix A
    print('')
    print('  * generating A')
    A = np.random.normal(0, 1, (m, n))
    # print('  * max_lam(AAt) / n = %.4e' % (largest_eigh(np.dot(A, A.transpose()), eigvals=(m - 1, m - 1))[0][0] / n))

    # compute true coefficients
    x_true = np.zeros((n,))
    x_true[0:non_zeros] = xstar

    # compute err variance and error
    err_sd = np.sqrt(np.var(np.dot(A, x_true)) / snr)
    err = err_sd * np.random.normal(0, 1, (m, ))

    # compute the response
    b = np.dot(A, x_true).reshape(m, ) + err
    # b = np.dot(A, x_true).reshape(m, )
    b += - np.mean(b)

    # find lam1_max, and determine lam1 and lam2
    Atb = np.dot(A.transpose(), b)
    lam1_max = LA.norm(Atb, np.inf) / alpha
    lam1 = alpha * c_lam * lam1_max
    lam2 = (1 - alpha) * c_lam * lam1_max

    # -------------------- #
    #  ssnal_elastic_core  #
    # -------------------- #

    print('')
    print('  * start ssnal_elastic')
    out_core = ssnal_elastic_core(A=A, b=b,
                                  lam1=lam1, lam2=lam2,
                                  x0=None, y0=None, z0=None, Aty0=None,
                                  sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
                                  step_reduce=step_reduce, mu=mu,
                                  tol_ssn=tol_ssn, tol_ssnal=tol_ssnal,
                                  maxiter_ssn=maxiter_ssn, maxiter_ssnal=maxiter_ssnal,
                                  use_cg=use_cg, r_exact=r_exact,
                                  print_lev=print_lev)

   




