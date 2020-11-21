"""Run solver on real data"""


import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import eigh as largest_eigh
from numpy import linalg as LA


from ssnal import ssnal_elastic_core

if __name__ == '__main__':

    # ----------------------------------------------------- #
    #  choose datasets to analyze and polynomial expansion  #
    # ----------------------------------------------------- #

    housing = True
    bodyfat = False
    traizines = False

    poly_degree = 8

    # --------------------- #
    #  set ssnal_parameter  #
    # --------------------- #

    # set c_lam
    c_lam = 0.6

    # set alpha
    alpha = 0.8

    # decide level of printing
    print_lev = 2

    # to choose between cg and exact method
    use_cg = True
    r_exact = 2000  # number of features such that we start using the exact method

    # to control sigma
    sgm = 0.005
    sgm_increase = 5
    sgm_change = 1

    # to select step size
    mu = 0.2
    step_reduce = 0.5

    # max iterations and tolerance
    maxiter_ssn = 40
    maxiter_ssnal = 100
    tol_ssn = 1e-6
    tol_ssnal = 1e-6

    # ------------------- #
    #  Uploading dataset  #
    # ------------------- #

    print('')
    print('  * creating polynomial expansion')
    if housing:
        from sklearn import datasets
        A, b = datasets.load_boston(return_X_y=True)

    if bodyfat:
        data = np.genfromtxt('./toy_data/bodyfat.csv',
                             delimiter=',', skip_header=True)
        A = np.column_stack((data[:, 0], data[:, 2:]))
        b = data[:, 1]

    if traizines:
        data = np.genfromtxt('./toy_data/traizines.csv',
                             delimiter=',', skip_header=True)
        A = data[:, 1:]
        b = data[:, 0]

    poly = PolynomialFeatures(poly_degree)
    A = poly.fit_transform(A)[:, 1:]
    A = (A - A.mean(axis=0)) / A.std(axis=0)
    b -= b.mean(axis=0)
    m, n = A.shape

    print('')
    print('  * dim A =', A.shape)
    print('  * max_lam(AAt) = %.4e' %
          largest_eigh(np.dot(A, A.T), eigvals=(m - 1, m - 1))[0][0])
    # TODO MM use np.linalg.norm(A, ord=2) ** 2?
    # TODO TB largest_eigh is more efficient. With a 1000x1000 matrix it took 0.3 sec while np.linalg.norm took 1.7sec
    print('')

    # find lam1_max, and determine lam1 and lam2
    lam1_max = LA.norm(A.T @ b, ord=np.inf) / alpha
    lam1 = alpha * c_lam * lam1_max
    lam2 = (1 - alpha) * c_lam * lam1_max

    # -------------------- #
    #  ssnal_elastic_core  #
    # -------------------- #

    print('')
    print('  * start ssnal_elastic')
    out_core = ssnal_elastic_core(
        A=A, b=b, lam1=lam1, lam2=lam2, x0=None, y0=None, z0=None, Aty0=None,
        sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
        step_reduce=step_reduce, mu=mu, tol_ssn=tol_ssn, tol_ssnal=tol_ssnal,
        maxiter_ssn=maxiter_ssn, maxiter_ssnal=maxiter_ssnal,
        use_cg=use_cg, r_exact=r_exact, print_lev=print_lev)
