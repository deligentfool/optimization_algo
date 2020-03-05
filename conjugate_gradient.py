from sympy import *
from utils import get_grad, get_hessian, get_norm, get_stagnation


def conjugate_gradient(params, func, init_values, stop_condition=1e-2):
    # * PRP
    values = Matrix(init_values)
    lam = Symbol('lam')
    beta = 0
    previous_d = 0
    previous_g = 0
    step = 0
    while True:
        g = get_grad(params, func)
        g = g.subs(dict(zip(params, list(values))))
        if get_norm(g) <= stop_condition:
            return list(values), func.subs(dict(zip(params, list(values))))
        if previous_g != 0:
            beta = (g.T * (g - previous_g)) / (get_norm(previous_g) ** 2)
            d = - g + beta[0] * previous_d
        else:
            d = - g
        lam_func = func.subs(dict(zip(params, list(values + lam * d))))
        lam_value = get_stagnation(lam_func)
        values = values + lam_value * d
        previous_d = d
        previous_g = g
        f_value = func.subs(dict(zip(params, list(values))))
        print('step: {}  params: {}  f: {}'.format(step, list(values), f_value))
        step += 1


if __name__ == '__main__':
    x1, x2 = symbols('x1, x2')
    f = (x1 ** 3 - x2) ** 2 + 2 * (x2 - x1) ** 4
    params, f_value = conjugate_gradient((x1, x2), f, [-0., -1.])
    print('final params: {}  final f value: {}'.format(params, f_value))