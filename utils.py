from sympy import *


def get_grad(params, func):
    grad_vec = []
    for param in params:
        grad = diff(func, param)
        grad_vec.append(grad)
    return Matrix(grad_vec)


def get_hessian(params, func):
    hessian_mat = []
    for param_i in params:
        hessian_mat.append([])
        for param_j in params:
            hessian_mat[-1].append(diff(diff(func, param_i), param_j))
    return Matrix(hessian_mat)


def get_stagnation(func):
    lam_func = diff(func)
    lam = solve(lam_func)
    return lam[0]


def get_norm(vec):
    norm = 0
    for i in range(len(vec)):
        norm += vec[i] ** 2
    return sqrt(norm)