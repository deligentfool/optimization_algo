from sympy import *
from utils import get_grad, get_hessian, get_norm, get_stagnation


def steepest_descent(params, func, init_values, stop_condition=1e-10):
    values = Matrix(init_values)
    lam = Symbol('lam')
    step = 0
    while True:
        g = get_grad(params, func)
        g = g.subs(dict(zip(params, list(values))))
        if get_norm(g) <= stop_condition:
            return list(values), func.subs(dict(zip(params, list(values))))
        lam_func = func.subs(dict(zip(params, list(values - lam * g))))
        lam_value = get_stagnation(lam_func)
        values = values - lam_value * g
        f_value = func.subs(dict(zip(params, list(values))))
        print('step: {}  params: {}  f: {}'.format(step, list(values), f_value))
        step += 1


if __name__ == '__main__':
    x1, x2 = symbols('x1, x2')
    f = (x1 ** 3 - x2) ** 2 + 2 * (x2 - x1) ** 4
    params, f_value = steepest_descent((x1, x2), f, [-0., -1.])
    print('final params: {}  final f value: {}'.format(params, f_value))