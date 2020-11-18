

# -------------------------------------------------------------- #
#                                                                #
#    auxiliary function for ssnal elastic net and SNALL lasso    #
#                                                                #
# -------------------------------------------------------------- #


import numpy as np
from numpy import linalg as LA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def prox_elst(v, par1, par2):

    """
    computes proximal operator for elastic net penalization

    """

    if par2 > 0:
        return 1 / (1 + par2) * (np.maximum(0, v - par1) - np.maximum(0, - v - par1))

    else:
        return np.maximum(0, v - par1) - np.maximum(0, - v - par1)


def prox_star_elst(v, par1, par2, t):

    """
    computes proximal operator for the conjugate of the elastic net penalization

    :param v: the argument is already divided by sigma: prox_(p*/sgm)(x/sgm = v)

    """

    return v - prox_elst(v * t, par1, par2) / t


def p_star_elst(v, par1, par2):

    """
    computes the conjugate function of the elastic net penalization

    """

    if par2 > 0:
        return np.sum(np.maximum(0, v - par1) ** 2 + np.maximum(0, - v - par1) ** 2) / (2 * par2)

    else:
        return np.sum(np.maximum(0, v - par1) ** 2 + np.maximum(0, - v - par1) ** 2)


def prim_obj_elst(A, x, b, lam1, lam2):

    """
    computes primal object of the elastic net

    """

    return 0.5 * LA.norm(np.dot(A, x) - b) ** 2 + lam1 * LA.norm(x, 1) + lam2 / 2 * LA.norm(x) ** 2


def dual_obj_elst(y, z, b, lam1, lam2):

    """
    computes dual object of the elastic net

    """

    return - (0.5 * LA.norm(y) ** 2 + np.dot(b.transpose(), y) + p_star_elst(z, lam1, lam2))


def phi_y_elst(y, x, b, Aty, sgm, lam1, lam2):

    """
    computes phi(y) for the elastic net problem

    """

    return (LA.norm(y) ** 2 / 2 + np.dot(b.transpose(), y) +
           (1 + sgm * lam2) / (2 * sgm) * LA.norm(prox_elst(x - sgm * Aty, sgm * lam1, sgm * lam2)) ** 2 -
            LA.norm(x) ** 2 / (2 * sgm))


def grad_phi_elst(A, y, x, b, Aty, sgm, lam1, lam2):

    """
    computes the gradient of phi(y) for the elastic net problem

    """

    return y + b - np.dot(A, prox_elst(x - sgm * Aty, sgm * lam1, sgm * lam2))


def factorization(A, sgm):

    """
    computes cholesky factorization of:
        I + sgm * A'A   if r > m
        I + sgm * AA'   if r<= m  (SWM formula)

    """

    m, r = A.shape
    At = A.transpose()

    if m >= r:  # special case, SWW formula
        A_star = 1 / sgm * np.eye(r) + np.dot(At, A)
    else:
        A_star = np.eye(m) + sgm * np.dot(A, At)

    L = LA.cholesky(A_star)
    Lt = L.transpose()

    return L, Lt


def plot_cv_ssnal_elstic(r_lm, ebic, gcv, cv, alpha, grid):

    """
    plots of: r_lm, ebic, gcv, cv for different values of alpha

    :param r_lm: list of r_lm. Each element of the list is the r_lm values for the respective alpha in alpha_list
    :param ebic: list of ebic. Each element of the list is the ebic values for the respective alpha in alpha_list
    :param gcv: list of gcv. Each element of the list is the gcv values for the respective alpha in alpha_list
    :param cv: list of cv. Each element of the list is the cv values for the respective alpha in alpha_list
    :param alpha: vec of different value of alpha considered
    :param grid: array of all the lambda1_ratio considered (same for all alphas)

    """

    # if the inputs are not list, we create them:
    if type(r_lm) != list:
        r_lm_list, ebic_list, gcv_list, cv_list = list(), list(), list(), list()
        r_lm_list.append(r_lm)
        ebic_list.append(ebic)
        gcv_list.append(gcv)
        cv_list.append(cv)
        alpha_vec = np.array([alpha])
        n_alpha = 1
    else:
        r_lm_list, ebic_list, gcv_list, cv_list, alpha_vec = r_lm, ebic, gcv, cv, alpha
        n_alpha = alpha.shape[0]

    # chech if we need to print cv
    if np.sum(cv_list[0]) == - cv_list[0].shape[0]:
        cv = False
    else:
        cv = True

    fig, ax = plt.subplots(2, 2)

    # ebic
    for i in range(n_alpha):
        indx = ebic_list[i] != -1
        t = grid[:ebic_list[i].shape[0]][indx]
        ax[0, 0].plot(t, ebic_list[i][indx], label=('alpha = %.2f' % alpha_vec[i]))
    ax[0, 0].legend(loc='best')
    ax[0, 0].set_title('ebic')
    ax[0, 0].set_xlim([grid[0], grid[-1]])

    # gcv
    for i in range(n_alpha):
        indx = gcv_list[i] != -1
        t = grid[:gcv_list[i].shape[0]][indx]
        ax[0, 1].plot(t, gcv_list[i][indx], label=('alpha = %.2f' % alpha_vec[i]))
    ax[0, 1].legend(loc='best')
    ax[0, 1].set_title('gcv')
    ax[0, 1].set_xlim([grid[0], grid[-1]])

    # r_lm
    for i in range(n_alpha):
        indx = r_lm_list[i] != -1
        t = grid[:r_lm_list[i].shape[0]][indx]
        ax[1, 0].plot(t, r_lm_list[i][indx], label=('alpha = %.2f' % alpha_vec[i]))
    ax[1, 0].legend(loc='best')
    ax[1, 0].set_title('selected features')
    ax[1, 0].set_xlim([grid[0], grid[-1]])

    # cv
    if cv:
        for i in range(n_alpha):
            indx = cv_list[i] != -1
            t = grid[:cv_list[i].shape[0]][indx]
            ax[1, 1].plot(t, cv_list[i][indx], label=('alpha = %.2f' % alpha_vec[i]))
        ax[1, 1].legend(loc='best')
        ax[1, 1].set_title('cross validation')
        ax[1, 1].set_xlim([grid[0], grid[-1]])

    plt.show()


