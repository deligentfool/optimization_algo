from sympy import *
from utils import get_grad, get_hessian, get_norm, get_stagnation
import numpy as np

def bisection(params, func, a, b, stop_condition=1e-2):
    a = Matrix(a)
    b = Matrix(b)
    step = 0
    while True:
        g = get_grad(params, func)
        g_a = g.subs(dict(zip(params, list(a))))
        g_b = g.subs(dict(zip(params, list(b))))
        assert g_a.values()[0] < 0
        assert g_b.values()[0] > 0
        bi = Matrix([(a[0] + b[0]) / 2])
        g_bi = g.subs(dict(zip(params, list(bi))))
        if g_bi.values()[0] > 0:
            b = bi
        else:
            a = bi
        print('step: {}  a: {}  b: {}'.format(step, list(a), list(b)))
        if np.abs(a[0] - b[0]) <= stop_condition:
            break
        step += 1


if __name__ == '__main__':
    x1 = symbols('x1')
    f = x1 ** 4 - 4 * x1 ** 3 - 6 * x1 ** 2 - 16 * x1 + 4
    bisection([x1], f, [0.] ,[6.], 1e-3)