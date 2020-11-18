

# -------------------------- #
#                            #
#    CV ssnal ELASTIC NET    #
#                            #
# -------------------------- #


import time
import numpy as np
from tqdm import tqdm
from numpy import linalg as LA
from sklearn.model_selection import KFold

from ssnal.auxiliary_functions import plot_cv_ssnal_elstic
from ssnal import ssnal_elastic_core


def ssnal_elastic_path(A, b,
                       c_lam_vec=None, alpha=0.9,
                       x0=None, y0=None, z0=None, Aty0=None,
                       max_selected=None,
                       cv=False, n_folds=10,
                       sgm=5e-3, sgm_increase=5, sgm_change=1,
                       step_reduce=0.5, mu=0.2,
                       tol_ssn=1e-6, tol_ssnal=1e-6,
                       maxiter_ssn=50, maxiter_ssnal=100,
                       use_cg=True, r_exact=2e4,
                       plot=False, print_lev=2):
    """
    --------------------------------------------------------------------------
    ssnal algorithm to solve the elastic net for a list of lambda1 and lambda2
    --------------------------------------------------------------------------

    ----------------------------------------------------------------------------------------------------------------------
    :param A: design matrix (m x n)
    :param b: response vector (m, )
    :param c_lam_vec: np.array to determine all the values of lambda1 -- lambda1 = c_lam_vec * lambda1_max
    :param alpha: an array to determin lambda 2 -- lam2 = (1 - alpha) * lam1
    :param max_selected: if given, the algorithm stops when selects a number of features > max_selected
    :param x0: initial value for the lagrangian multiplier (variable of the primal problem) (n, ) -- vector 0 if not given
    :param y0: initial value fot the first variable of the dual problem  (m, ) -- vector of 0 if not given
    :param z0: initial value for the second variable of the dual problem (n, ) -- vector of 0 if not given
    :param Aty0: np.dot(A.transpose(), y0) (n,)
    :param max_selected: maximum number of features selected
    :param cv: True/False. I true, a cross validation is performed
    :param n_folds: number of folds to perform the cross validation
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
    :param print_lev: different level of printing (0, 1, 2, 3, 4)
    :param plot: True/False. If true a plot of r_lm, gcv, extended bic and cv (if cv == True) is displayed
    ----------------------------------------------------------------------------------------------------------------------

    ---------------------------------------------------------------------------------------
    :return[0] aic_vec: aic values for each lambda1
    :return[1] ebic_vec: ebic values for each lambda1
    :return[2] gcv_vec: gcv values for each lambda1
    :return[3] cv_vec: cv values for each lambda1
    :return[4] r_vec: number of selected features for each lambda1
    :return[5] r_lm_vec: number of selected features by the debiased model for each lambda1
    :return[6] iter_vec: number of ssnal iterations for each lambda1
    :return[7] times_vec: ssnal time for each lambda1
    :return[8] full_time: time for fitting the full lambda for all the sequence of lambda1
    :return[9] cv_time: time for cross validation
    :return[10] total_time: total time
    ---------------------------------------------------------------------------------------

    REMEMBER: the output os ssnal_elastic_core is a list which contains:
    --------------------------------------------------------------------------------------------------
    :return[0] x: optimal value of the primal variable
    :return[1] y: optimal value of the first dual variable
    :return[2] z: optimal value of the second dual variable
    :return[3] x_lm: linear regression estimates on the
    :return[4] r: number of selected features
    :return[5] r_lm: number of selected features by the debiased model
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

    lam1_max = LA.norm(np.dot(A.transpose(), b), np.inf) / alpha

    if x0 is None:
        x0 = np.zeros((n,))
    if y0 is None:
        y0 = np.zeros((m,))
    if z0 is None:
        z0 = np.zeros((n,))

    if max_selected is None:
        max_selected = n

    if c_lam_vec is None:
        c_lam_vec = np.logspace(start=3, stop=1, num=100, base=10.0)/1000

    n_lam1 = c_lam_vec.shape[0]

    sgm0 = sgm

    # ---------------------- #
    #    initialize flags    #
    # ---------------------- #

    convergence = True
    reached_max = False

    n_lam1_stop = n_lam1

    # ---------------------------- #
    #    create output matrices    #
    # ---------------------------- #

    aic_vec, ebic_vec, gcv_vec = - \
        np.ones([n_lam1]), - np.ones([n_lam1]), - np.ones([n_lam1])
    times_vec, r_vec, r_lm_vec, iter_vec = - \
        np.ones([n_lam1]), - np.ones([n_lam1]), - \
        np.ones([n_lam1]), - np.ones([n_lam1])

    # ---------------------- #
    #    solve full model    #
    # ---------------------- #

    if print_lev > 0:

        print()
        print('--------------------------------------------------')
        print('  solving full model  ')
        print('--------------------------------------------------')

    start_full = time.time()

    for i in range(n_lam1):

        lam1 = alpha * c_lam_vec[i] * lam1_max
        lam2 = (1 - alpha) * c_lam_vec[i] * lam1_max

        if print_lev > 2:
            print('--------------------------------------------------------------------------------------------------')
            print(' FULL MODEL:  lambda1 (ratio) = %.2e  (%.2e)  |  lambda2 = %.2e  |  sigma0 = %.2e' % (
                lam1, c_lam_vec[i], lam2, sgm))
            print('--------------------------------------------------------------------------------------------------')

        # ------------------- #
        #    perform ssnal    #
        # ------------------- #

        fit = ssnal_elastic_core(A=A, b=b, lam1=lam1, lam2=lam2,
                                 x0=x0, y0=y0, z0=z0, Aty0=Aty0,
                                 sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
                                 step_reduce=step_reduce, mu=mu,
                                 tol_ssn=tol_ssn, tol_ssnal=tol_ssnal,
                                 maxiter_ssn=maxiter_ssn, maxiter_ssnal=maxiter_ssnal,
                                 use_cg=use_cg, r_exact=r_exact,
                                 print_lev=print_lev - 3)

        # ----------------------- #
        #    check convergence    #
        # ----------------------- #

        if not fit[10]:
            convergence = False
            break

        # ---------------------------- #
        #    update starting values    #
        # ---------------------------- #

        x0, y0, z0, Aty0, sgm = fit[0], fit[1], fit[2], fit[15], fit[9]

        # ---------------------------- #
        #    update output matrices    #
        # ---------------------------- #

        times_vec[i], r_vec[i], r_lm_vec[i], iter_vec[i] = fit[13], fit[4], fit[5], fit[14]

        r_lm = fit[5]
        if r_lm > 0:
            aic_vec[i], ebic_vec[i], gcv_vec[i] = fit[6], fit[7], fit[8]

        # --------------------------------------- #
        #    check number of selected features    #
        # --------------------------------------- #

        r_lm = fit[5]
        if r_lm > max_selected:
            n_lam1_stop = i + 1
            reached_max = True
            break

    # ------------------- #
    #    end full model   #
    # ------------------- #

    full_time = time.time() - start_full

    if not convergence:
        print('--------------------------------------------------')
        print(' snall has not converged for lam1 = %.4f, lam2 = %.4f ' %
              (lam1, lam2))
        print('--------------------------------------------------')

    if reached_max:
        print('--------------------------------------------------')
        print(' max number of features has been selected')
        print('--------------------------------------------------')

    # -------------- #
    #    start cv    #
    # -------------- #

    cv_time = 0
    cv_mat = - np.ones([n_lam1_stop, n_folds])

    if cv and convergence:

        print('--------------------------------------------------')
        print('  performing cv  ')
        print('--------------------------------------------------')

        x0_cv, z0_cv = np.zeros((n,)), np.zeros((n,))
        Aty0_cv = None
        sgm_cv = sgm0

        fold = 0

        start_cv = time.time()

        # ------------- #
        #    split A    #
        # ------------- #

        kf = KFold(n_splits=n_folds)
        kf.get_n_splits(A)

        # -------------------- #
        #    loop for folds    #
        # -------------------- #

        for train_index, test_index in kf.split(A):

            A_train, A_test = A[train_index], A[test_index]
            b_train, b_test = b[train_index], b[test_index]

            y0_cv = np.zeros((np.shape(train_index)[0],))

            # -------------------- #
            #    loop for lam1    #
            # -------------------- #

            for i_cv in tqdm(range(n_lam1_stop)):

                lam1 = c_lam_vec[i_cv] * lam1_max
                lam2 = (1 - alpha) * lam1

                # ------------------- #
                #    perform ssnal    #
                # ------------------- #

                fit_cv = ssnal_elastic_core(A=A_train, b=b_train, lam1=lam1, lam2=lam2,
                                            x0=x0_cv, y0=y0_cv, z0=z0_cv, Aty0=Aty0_cv,
                                            sgm=sgm_cv, sgm_increase=sgm_increase, sgm_change=sgm_change,
                                            step_reduce=step_reduce, mu=mu,
                                            tol_ssn=tol_ssn, tol_ssnal=tol_ssnal,
                                            maxiter_ssn=maxiter_ssn, maxiter_ssnal=maxiter_ssnal,
                                            use_cg=use_cg, r_exact=r_exact,
                                            print_lev=0)

                # ------------------- #
                #    update cv mat    #
                # ------------------- #

                cv_mat[i_cv, fold] = LA.norm(
                    np.dot(A_test, fit_cv[3]) - b_test) ** 2

                # ---------------------------- #
                #    update starting values    #
                # ---------------------------- #

                if i_cv == n_lam1_stop:
                    x0_cv, y0_cv, z0_cv, Aty0_cv, sgm_cv = None, None, None, None, sgm0

                else:
                    x0_cv, y0_cv, z0_Cv, Aty0_cv, sgm_cv = fit_cv[
                        0], fit_cv[1], fit_cv[2], fit_cv[15], fit_cv[9]

            # ------------------------ #
            #    end loop for lam1    #
            # ------------------------ #

            fold += 1

        # ------------ #
        #    end cv    #
        # ------------ #

        cv_time = time.time() - start_cv

    # ---------------------------- #
    #    printing final results    #
    # ---------------------------- #

    if cv:
        cv_vec = cv_mat.mean(1) / m
    else:
        cv_vec = - np.ones([n_lam1_stop])

    total_time = full_time + cv_time

    time.sleep(0.1)

    print('')
    print('')
    print('------------------------------------------------------------')
    print(' total time:  %.4f' % total_time)

    if cv:
        print('------------------------------------------------------------')
        print('  full time:  %.4f' % full_time)
        print('------------------------------------------------------------')
        print('  cv time:    %.4f' % cv_time)

    print('------------------------------------------------------------')

    if print_lev > 1:

        print('')
        # print('------------------------------------------------------------')
        print('    c_lam    lam1      lam2    r_lm    gcv     ebic     cv      ')
        print('------------------------------------------------------------')

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print(np.stack((c_lam_vec[:n_lam1_stop],
                        alpha * c_lam_vec[:n_lam1_stop] * lam1_max,
                        (1 - alpha) * c_lam_vec[:n_lam1_stop] * lam1_max,
                        r_lm_vec[:n_lam1_stop],
                        gcv_vec[:n_lam1_stop],
                        ebic_vec[:n_lam1_stop],
                        cv_vec), -1))
        print('\n')

    if plot:
        plot_cv_ssnal_elstic(r_lm_vec, ebic_vec, gcv_vec,
                             cv_vec, alpha, c_lam_vec)

    return aic_vec[:n_lam1_stop], \
        ebic_vec[:n_lam1_stop], \
        gcv_vec[:n_lam1_stop], \
        cv_vec, \
        r_vec[:n_lam1_stop], \
        r_lm_vec[:n_lam1_stop], \
        iter_vec[:n_lam1_stop], \
        times_vec[:n_lam1_stop], \
        full_time, cv_time, total_time
