from sympy import *
from utils import get_grad, get_hessian, get_norm, get_stagnation


def newton(params, func, init_values, stop_condition=1e-2):
    values = Matrix(init_values)
    step = 0
    while True:
        g = get_grad(params, func)
        g = g.subs(dict(zip(params, list(values))))
        if get_norm(g) <= stop_condition:
            return list(values), func.subs(dict(zip(params, list(values))))
        h = get_hessian(params, func)
        h = h.subs(dict(zip(params, list(values))))
        values = values - h ** (-1) * g
        f_value = func.subs(dict(zip(params, list(values))))
        print('step: {}  params: {}  f: {}'.format(step, list(values), f_value))
        step += 1


if __name__ == '__main__':
    x1, x2 = symbols('x1, x2')
    f = (x1 ** 3 - x2) ** 2 + 2 * (x2 - x1) ** 4
    params, f_value = newton((x1, x2), f, [-0., -1.])
    print('final params: {}  final f value: {}'.format(params, f_value))