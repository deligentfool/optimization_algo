from math import *
import matplotlib.pyplot as plt  # 绘图模块
from pylab import *  # 绘图辅助模块


def Fab_list(Fmax):
    a, b = 1, 1
    Fablist = [a, b]
    while Fablist[-1] < Fmax:
        a, b = b, a + b
        Fablist.append(b)
    return Fablist


def fab_method(F, a_0, b_0, delta, epsilon=0.01):
    Fmax = (b_0 - a_0) / delta
    fab_list = Fab_list(Fmax)
    a = a_0
    b = b_0
    for k in range(1, len(fab_list) - 1):
        t_k = b + fab_list[-k-1] / fab_list[-k] * (a - b)
        t_k_prime = a + fab_list[-k-1] / fab_list[-k] * (b - a)
        #assert t_k < t_k_prime
        if t_k_prime == t_k:
            t_k_prime = t_k + epsilon
        phi_t_k = F(t_k)
        phi_t_k_prime = F(t_k_prime)
        if phi_t_k < phi_t_k_prime:
            b = t_k_prime
        else:
            a = t_k
        print('k:{}\ta:{:.4f}\tb:{:.4f}\tt_k:{:.4f}\tt_k_prime:{:.4f}\tphi_t_k:{:.4f}\tphi_t_k_prime:{:.4f}'.format(k, a, b, t_k, t_k_prime, phi_t_k, phi_t_k_prime))

def gold_method(F, a_0, b_0, delta):
    n = 1
    while True:
        if 0.618 ** (n - 1) * (b_0 - a_0) <= delta:
            break
        n += 1
    a = a_0
    b = b_0
    for k in range(n):
        t_k = b + 0.618 * (a - b)
        t_k_prime = a + 0.618 * (b - a)
        phi_t_k = F(t_k)
        phi_t_k_prime = F(t_k_prime)
        if phi_t_k < phi_t_k_prime:
            b = t_k_prime
        else:
            a = t_k
        print('k:{}\ta:{:.4f}\tb:{:.4f}\tt_k:{:.4f}\tt_k_prime:{:.4f}\tphi_t_k:{:.4f}\tphi_t_k_prime:{:.4f}'.format(k, a, b, t_k, t_k_prime, phi_t_k, phi_t_k_prime))


if __name__ == '__main__':
    F = lambda x: 3 * x ** 2 -21.6 * x -1
    #fab_method(F, 0, 25, (25 - 0) * 0.08)
    gold_method(F, 0, 25, (25 - 0) * 0.08)