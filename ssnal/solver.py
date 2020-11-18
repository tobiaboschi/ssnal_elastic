

# --------------------------- #
#                             #
#    ssnAL ELASTIC NET CORE   #
#                             #
# --------------------------- #


import time
import numpy as np
from numpy import linalg as LA
import scipy.sparse.linalg as s_LA
from sklearn.linear_model import LinearRegression

import ssnal.auxiliary_functions as AF


def ssnal_elastic_core(A, b,
                       lam1, lam2,
                       x0=None, y0=None, z0=None, Aty0=None,
                       sgm=5e-3, sgm_increase=5, sgm_change=3,
                       step_reduce=0.5, mu=0.2,
                       tol_ssn=1e-6, tol_ssnal=1e-6,
                       maxiter_ssn=50, maxiter_ssnal=100,
                       use_cg=False, r_exact=2e4,
                       print_lev=1):

    """
    --------------------------------------------------------------------------------
    ssnal algorithm to solve the elastic net for fixed values of lambda1 and lambda2
    --------------------------------------------------------------------------------

    ----------------------------------------------------------------------------------------------------------------------
    :param A: design matrix (m x n)
    :param b: response vector (m, )
    :param lam1: lasso penalization
    :param lam2: ridge penalization, if = 0, the function perform the ssnal for lasso
    :param x0: initial value for the lagrangian multiplier (variable of the primal problem) (n, ) -- vector 0 if not given
    :param y0: initial value fot the first variable of the dual problem  (m, ) -- vector of 0 if not given
    :param z0: initial value for the second variable of the dual problem (n, ) -- vector of 0 if not given
    :param Aty0: np.dot(A.transpose(), y0) (n,)
    :param sgm: starting value of the augmented lagrangian parameter sigma
    :param sgm_increase: increasing factor of sigma
    :param sgm_change: we increase sgm -- sgm *= sgm_increase -- every sgm_change iterations
    :param step_reduce: dividing factor of the step size during the linesearch
    :param mu: multiplicative factor fot the lhs of the linesearch condition
    :param tol_ssn: tolerance for the ssn algorithm
    :param tol_ssnal: global tolerance of the ssnal algorithm
    :param maxiter_ssn: maximum number of iterations for ssn
    :param maxiter_ssnal: maximum number of global iterations
    :param use_cg: True/False. If true, the conjugate gradient method is used to find the direction of the ssn
    :param r_exact: number of features such that we start using the exact method
    :param print_lev: different level of printing (0, 1, 2)
    ----------------------------------------------------------------------------------------------------------------------

    --------------------------------------------------------------------------------------------------
    :return[0] x: optimal value of the primal variable
    :return[1] y: optimal value of the first dual variable
    :return[2] z: optimal value of the second dual variable
    :return[3] x_lm: linear regression estimates on the
    :return[4] r: number of selected features
    :return[5] r_lm: number of selected features in the debiased model
    :return[6] aic: aic computed on the debiased estimates
    :return[7] ebic: extended bic computed on the debiased estimates
    :return[8] gcv: gcv computed on the debiased estimates
    :return[9] sgm: final value of the augmented lagrangian parameter sigma
    :return[10] lam1: lasso penalization
    :return[11] lam2: ridge penalization
    :return[12] convergence_ssnal: True/False. If true, the ssnal has converged
    :return[13] ssnal_time: total time of ssnal
    :return[14] it_ssnal: total ssnal's iteration
    :return[15] Aty: np.dot(A.transpose(), y) computed at the optimal y. Useful to implement warmstart
    --------------------------------------------------------------------------------------------------

    """

    # -------------------------- #
    #    initialize variables    #
    # -------------------------- #

    m, n = A.shape

    x = x0
    y = y0
    z = z0
    Aty = Aty0

    if x is None:
        x = np.zeros((n,))
    if y is None:
        y = np.zeros((m,))
    if z is None:
        z = np.zeros((n,))
    if Aty0 is None:
        Aty = np.dot(A.transpose(), y)

    convergence_ssnal = False

    # ---------------------------- #
    #    start ssnal iterations    #
    # ---------------------------- #

    start_ssnal = time.time()

    for it_ssnal in range(maxiter_ssnal):

        if print_lev > 1:
            print('')
            print('  ssnal iteration = %.f  |  sgm = %.2e' % (it_ssnal + 1, sgm))
            print('  -------------------------------------------------------------------')

        # --------------- #
        #    start ssn    #
        # --------------- #

        start_ssn = time.time()

        convergence_ssn = False
        x_tilde = x - sgm * Aty

        for it_ssn in range(maxiter_ssn):

            # ------------- #
            #    find AJ    #
            # ------------- #

            indx = (np.absolute(x_tilde) > sgm * lam1).reshape(n)
            xJ = x[indx]
            AJ = A[:, indx]
            AJty = np.dot(AJ.transpose(), y)

            m, r = AJ.shape

            correction = 1 / (1 + sgm * lam2)

            # ------------------------- #
            #    compute direction d    #
            # ------------------------- #

            rhs = - AF.grad_phi_elst(AJ, y, xJ, b, AJty, sgm, lam1, lam2)
            # rhs = - grad_phi(A, y, x, b, Aty, sgm, lam1, lam2)

            if r == 0:
                method = 'E '
                d = rhs

            else:

                # ------------------------ #
                #    conjugate gradient    #
                # ------------------------ #

                if r > r_exact and use_cg:  # and kkt3 > 1e-2
                    method = 'CG'
                    if m >= r:
                        # special case, SMW formula
                        rhs_temp = np.dot(AJ.transpose(), rhs)
                        A_star = 1 / (sgm * correction) * np.eye(r) + np.dot(AJ.transpose(), AJ)
                        d_temp = s_LA.cg(A_star, rhs_temp, tol=1e-04, maxiter=1000)[0]
                        d = rhs - np.dot(AJ, d_temp)
                    else:
                        A_star = np.eye(m) + sgm * correction * np.dot(AJ, AJ.transpose())
                        d = s_LA.cg(A_star, rhs, tol=1e-04, maxiter=1000)[0]

                # ---------------- #
                #   exact method   #
                # ---------------- #

                else:
                    method = 'E '
                    LJ, LJt = AF.factorization(AJ, sgm * correction)
                    if m >= r:
                        # special case, SMW formula
                        d_temp = LA.solve(LJt, LA.solve(LJ, np.dot(AJ.transpose(), rhs)))
                        d = rhs - np.dot(AJ, d_temp)

                        # d_temp = LA.solve(np.eye(r) / (sgm * correction) + np.dot(AJ.T, AJ), np.dot(AJ.T, rhs))
                        # d = (rhs - np.dot(AJ, d_temp))
                    else:
                        d = LA.solve(LJt, LA.solve(LJ, rhs))

                        # A_star = np.eye(m) + sgm * correction * np.dot(AJ, AJ.transpose())
                        # d = LA.solve(A_star, rhs)

            # ------------------------------ #
            #    linesearch for step size    #
            # ------------------------------ #

            step_size = 1

            rhs_term_1 = AF.phi_y_elst(y, xJ, b, AJty, sgm, lam1, lam2)
            rhs_term_2 = np.dot(-rhs.transpose(), d)
            # rhs_term_1 = phi_y_elst(y, x, b, Aty, sgm, lam1, lam2)

            while True:
                y_new = y + step_size * d
                Aty_new = np.dot(A.transpose(), y_new)
                if AF.phi_y_elst(y_new, x, b, Aty_new, sgm, lam1, lam2) <= rhs_term_1 + mu * step_size * rhs_term_2:
                    break
                step_size *= step_reduce

            # ---------------------- #
            #    update variables    #
            # ---------------------- #

            y = y_new
            Aty = Aty_new

            z = AF.prox_star_elst(x / sgm - Aty, sgm * lam1, sgm * lam2, sgm)
            x_tilde = x - sgm * Aty
            x_temp = x_tilde - sgm * z

            # --------------------------- #
            #    ssn convergence check    #
            # --------------------------- #

            if r > 0:
                kkt1 = LA.norm(np.dot(AJ, x_temp[indx]) - b - y) / (1 + LA.norm(b))
            else:
                kkt1 = LA.norm(np.dot(A, x_temp) - b - y) / (1 + LA.norm(b))

            if print_lev > 1:
                if it_ssn + 1 > 9:
                    space = ''
                else:
                    space = ' '
                print(space, '  %.f| ' % (it_ssn + 1),  method, ' kkt1 = %.2e  -  step_size = %.1e  -  r = %.f' % (kkt1, step_size, r), sep='')

            if kkt1 < tol_ssn or r == 0:
                convergence_ssn = True
                break

        # ------------- #
        #    end ssn    #
        # ------------- #

        time_ssn = time.time() - start_ssn

        if print_lev > 1:
            print('  -------------------------------------------------------------------')
            print('  ssn time = %.4f' % time_ssn)
            print('  -------------------------------------------------------------------')

        if not convergence_ssn:
            print('\n \n')
            print('  * ssn DOES NOT CONVERGE: try to increase the number of ssn iterations or to reduce the lambdas')
            break

        # ----------------------------- #
        #    ssnal convergence check    #
        # ----------------------------- #

        x = x_temp
        xJ = x[indx]
        zJ = z[indx]

        # compute kkt3
        kkt3 = LA.norm(z + Aty) / (1 + LA.norm(z) + LA.norm(y))

        # compute objective functions
        prim = AF.prim_obj_elst(AJ, xJ, b, lam1, lam2)
        dual = AF.dual_obj_elst(y, zJ, b, lam1, lam2)
        dual_gap = np.abs(prim - dual)/(prim + dual)

        if print_lev > 1:
            print('  kkt3 = %.5e  -  dual gap = %.5e' % (kkt3, dual_gap))

        if kkt3 < tol_ssnal and dual_gap < tol_ssnal:
            convergence_ssnal = True
            it_ssnal += 1
            break

        if np.mod(it_ssnal + 1, sgm_change) == 0:
            sgm *= sgm_increase

        # if kkt1 < 1e-6:
        #     tol_ssn = max(tol_ssn, 1e-6)
        # else:
        #     tol_ssn *= 1 * kkt1

    # --------------- #
    #    end ssnal    #
    # --------------- #

    ssnal_time = time.time() - start_ssnal

    # -------------------------------- #
    #    debias and model selection    #
    # -------------------------------- #

    aic, ebic, gcv = None, None, None

    if r > 0:

        # fit linear regression
        lm = LinearRegression().fit(AJ, b)

        # find the non_zeros of the debiased model
        lm_indx = np.abs(lm.coef_) > 1e-6
        x_lm_temp = lm.coef_[lm_indx]
        r_lm = x_lm_temp.shape[0]

        # subselect AJ
        A_lm = AJ[:, lm_indx]

        # ccompute res
        res = b - np.dot(A_lm, x_lm_temp)

        # compute dof
        df_core = LA.inv(np.dot(A_lm.transpose(), A_lm) + lam2 * np.eye(r_lm))
        df = np.trace(np.dot(np.dot(A_lm, df_core), A_lm.transpose()))

        # compute rss
        rss = LA.norm(res) ** 2

        # compute model selection criteri
        ebic = np.log(rss / m) + df * np.log(m) / m + df * np.log(n) / m
        gcv = rss / m / (1 - df / m) ** 2

        # create final debiased solution of lenght n
        x_lm = np.zeros(n, )
        indx[indx] = lm_indx
        x_lm[indx] = x_lm_temp

    else:

        x_lm = x
        r_lm = 0

    # ---------------------------- #
    #    printing final results    #
    # ---------------------------- #

    if convergence_ssn:

        if print_lev > 0:

            print('')
            print('  ==================================================')
            print('   * iterations ......... %.f' % it_ssnal)
            print('   * ssnal time ......... %.4f' % ssnal_time)
            print('   * prim object ........ %.4e' % prim)
            print('   * dual object ........ %.4e' % dual)
            print('   * kkt3 ............... %.4e' % kkt3)
            print('   * min(x) ............. %.4e' % min(x))
            print('   * max(x) ............. %.4e' % max(x))
            # print('   * nonzeros in x ...... %.f' % r)
            print('   * nonzeros in x_lm ... %.f' % r_lm)

            if r > 0:
                print('   * ebic ............... %.4f' % ebic)
                print('   * gcv ................ %.4f' % gcv)

            print('  ==================================================')
            print('\n')

        if not convergence_ssnal:
            print('\n')
            print('   * ssnal HAS NOT CONVERGED:')
            print('     (try to increase the number of iterations)')
            print('\n')

    return x, y, z, x_lm, r, r_lm, aic, ebic, gcv, sgm, lam1, lam2, convergence_ssnal, ssnal_time, it_ssnal, Aty

