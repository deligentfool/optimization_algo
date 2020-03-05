from sympy import *
from utils import get_grad, get_hessian, get_norm, get_stagnation


def quasi_newton(params, func, init_values, stop_condition=1e-5):
    # * BFGS
    values = Matrix(init_values)
    lam = Symbol('lam')
    next_g = 0
    next_values = 0
    h = eye(len(params))
    step = 0
    while True:
        g = get_grad(params, func)
        g = g.subs(dict(zip(params, list(values))))
        d = - h ** (-1) * g
        lam_func = func.subs(dict(zip(params, list(values + lam * d))))
        lam_value = get_stagnation(lam_func)
        next_values = values + lam_value * d
        if get_norm(g) <= stop_condition:
            return list(values), func.subs(dict(zip(params, list(next_values))))
        else:
            next_g = get_grad(params, func)
            next_g = next_g.subs(dict(zip(params, list(next_values))))
            s = next_values - values
            y = next_g - g
            h = (eye(len(params)) - (s * y.T) / (s.T * y)[0]) * h * (eye(len(params)) - (s * y.T) / (s.T * y)[0]).T + (s * s.T) / (s.T * y)[0]
        values = next_values
        f_value = func.subs(dict(zip(params, list(values))))
        print('step: {}  params: {}  f: {}'.format(step, list(values), f_value))
        step += 1


if __name__ == '__main__':
    x1, x2 = symbols('x1, x2')
    f = (x1 ** 3 - x2) ** 2 + 2 * (x2 - x1) ** 4
    params, f_value = quasi_newton((x1, x2), f, [-0., -1.])
    print('final params: {}  final f value: {}'.format(params, f_value))