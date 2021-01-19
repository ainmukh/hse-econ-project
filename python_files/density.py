import sympy
import numpy as np
import pandas as pd
import scipy.integrate as integrate
from symengine import Symbol, pi, Float, diff, log, sin, exp  # , cos, Add,
from scipy.optimize import approx_fprime, minimize as mini, line_search
# from math import exp, sin
import matplotlib.pyplot as plt
from scipy.stats import norm, logistic, expon, uniform
import math
import plotly_express as px


def taylor(func, x0, t, eps=52, k=None, drvs=None):
    x = Symbol('x')
    if not drvs:
        k, drvs = 0, [func] + [0] * t
    s = func.subs({x: Float(x0, dps=eps)})
    for i in range(1, t):
        if not drvs[k + i]:
            drvs[k + i] = sympy.gcd_terms(diff(drvs[k + i - 1], x))
        s += drvs[k + i].subs({x: Float(x0, dps=eps)}) / (sympy.factorial(i)) * (x - x0) ** i
    return s


class Logistic:

    # set parameters of logistic distribution
    # eps sets the precision of a symengine Float
    # eps is used when substitute symbolic variables with float
    def __init__(self, alpha=0, beta=1, k=1, eps=52):
        self.loc = alpha  # location
        self.scale = beta
        self.shape = k
        self.eps = eps

    # logistic probability density function f(x)
    def pdf(self, x):
        return self.shape * exp((self.loc - x) / self.scale) /\
               (self.scale * (1 + exp((self.loc - x) / self.scale))**(self.shape + 1))

    # cumulative distribution function F(x)
    def cdf(self, x):
        return 1 / (1 + exp((self.loc - x) / self.scale))**self.shape

    # moment generating function of a logistic distribution – M_Y(t)
    # n is the number of samples for monte carlo integration
    def mgf(self, t):
        j = Symbol('j')
        return exp(self.loc * t) * pi * \
            sympy.product((self.scale * t + j), (j, 0, self.shape - 1)) / \
            (sin(pi * self.scale * t) * sympy.factorial(self.shape - 1))

    # (raw) moment of a logistic distribution
    # k – moment ordinal
    # MGF and its derivatives are not defined at x = 0, thus i use h
    # tay is for number of Taylor series terms used to numerically calculate the MGFs derivative at x = 0
    # to optimize moment calculating i store all the derivatives
    # storing works when calculating ith moments for i in range
    # differentiating for derivatives array in taylor
    def moment(self, k, h=1e-4, tay=10,
               store=False, derivatives=None,
               truncated=False, t_1=float('-inf'), t_2=float('inf')):

        if k % 2 == 1 and not truncated:  # odd moments are 0 and im not sure about truncated NEED TO TEST
            return 0.

        if truncated:
            f = integrate.quad(lambda y: y ** k * self.pdf(y), t_1, t_2)[0]
            f = f / (self.cdf(t_2) - self.cdf(t_1))

        else:
            x = Symbol('x')
            if store:
                f = derivatives[k]  # take a func
                f = sympy.collect(f, x)  # factorize expression – reduces inaccuracy
                f = taylor(f, h, tay, self.eps, k, derivatives)  # Taylor expansion
            else:
                f = diff(self.mgf(x), *[x] * k)  # differentiate k times with respect to x
                f = sympy.factor(f) if not truncated else f  # as collect but somehow gives better accuracy, worse speed
                f = taylor(f, h, tay, self.eps)
            f = f.subs({x: Float(0, dps=self.eps)})
        return f.evalf(15)


class Approximant(Logistic):
    # initialize approximate distribution
    def __init__(self, alpha=0, beta=1, k=1):
        Logistic.__init__(self, alpha=alpha, beta=beta, k=k)

    # approximate pdf
    # x is A POINT
    # a is a set of constants of a pdf
    # moments may be given
    # n is for monte carlo
    def a_pdf(self, x, a=None,
              moments_calculated=False, moments=None,
              h=1e-2, tay=10, truncated=False, t_1=float('-inf'), t_2=float('inf')):

        # gamma
        powers = pd.Series(range(a.size))
        vec_pow = np.vectorize(pow, signature='(),(k)->(k)')

        def df_pow(v1, v2):
            return vec_pow(v1, v2)

        vec_top = np.vectorize(df_pow, signature='(n),(k)->(n,k)')
        x_p = pd.DataFrame(vec_top(x.values, powers)).transpose()
        s_1 = a.dot(x_p)

        # phi
        a_in = pd.Series(range(a.size))
        # a_list = a.to_list()
        a_list = a.tolist()
        k = a.size
        a_in = a_in.apply(lambda v: pd.Series([0] * v + a_list + [0] * (k - 1 - v)))
        a_sq = a.dot(a_in)

        if not moments_calculated:
            if not truncated:
                y = Symbol('y')
                moments = pd.Series(range(2 * k - 1))
                d = [self.mgf(y)] + [0] * (2 * k - 2 + 10)  # storing derivatives
                for i in range(2 * k - 1):
                    moments.append(self.moment(i, h, tay, store=True, derivatives=d))

            else:
                moments = pd.Series(range(2 * k - 1))
                mom = np.vectorize(self.moment)
                moments = mom(moments.values, truncated=truncated, t_1=t_1, t_2=t_2)

        s_2 = a_sq.dot(moments)

        # define pdf
        f = np.vectorize(Logistic.pdf)
        f = pd.Series(s_1**2 * f(self, x.values) / s_2)

        return f

    # a is alpha weights, x – sample
    # m is for moments
    # n is sample size
    def log_likelihood(self, a, x, m):
        logarithm = np.vectorize(log)
        p = logarithm(self.a_pdf(x, a, moments_calculated=True, moments=m).values)
        p = p.astype('float64')
        # e = pd.Series([1] * n)
        # p = e.dot(p)
        p = p.sum()
        # print(type(p), p)
        return -1 * p


# optimizer
class Optimize(Approximant):
    # f – function of the class, a – alpha, x – sample
    # eta – learning rate
    def gd(self, f, a, x,
           truncated=False,
           eta=3e-4,
           minimize=True,
           graph=True):

        k = a.size
        # x = x.sort_values()
        eta *= [-1, 1][minimize]

        # calculate moments
        m = pd.Series(range(2 * k - 1))
        moment = np.vectorize(self.moment)
        m = moment(m.values, truncated=truncated, t_1=x.min(), t_2=x.max())
        m = pd.Series(m)

        a = mini(f, a, (x, m), method='BFGS')['x']
        print(a)

        # Adam
        # b1, b2, eps = 0.9, 0.999, 1e-8
        # for j in range(epoch):
        #     m_prev, v_prev = 0, 0
        #     for i in range(iteration):
        #         x_s = x.sample(n=5000)
        #         gradient = approx_fprime(a, f, np.repeat(1e-8, a.size, axis=0), x_s, m)
        #         m_curr = (b1 * m_prev + (1 - b1) * gradient) / (1 - b1**(i + 1))
        #         v_curr = (b2 * v_prev + (1 - b2) * gradient**2) / (1 - b2**(i + 1))
        #         # print((b1 * m_prev + (1 - b1) * gradient))
        #         a = a - eta * (m_curr / (np.sqrt(v_curr) + eps))
        #
        #         m_prev, v_prev = m_curr, v_curr

        # B F G S
        # def prime(xk, args=(), f0=None):
        #     function = f
        #     epsilon = np.repeat(1e-8, xk.size, axis=0)
        #     if f0 is None:
        #         f0 = function(*((xk,) + args))
        #     gr = np.zeros((len(xk),), float)
        #     ei = np.zeros((len(xk),), float)
        #     for j in range(len(xk)):
        #         ei[j] = 1.0
        #         d = epsilon * ei
        #         gr[j] = (function(*((xk + d,) + args)) - f0) / d[j]
        #         ei[j] = 0.0
        #     return gr

        # eps = 1e-5
        # h = np.eye(k)
        # i = 0
        # grad = approx_fprime(a, f, np.repeat(1e-8, a.size, axis=0), x, m)
        # while v_norm(grad) > eps:
        #     p = -1 * h.dot(grad)
        #     alpha = line_search(f, prime, a, p, args=(x, m))[0]
        #     if not alpha:
        #         alpha = 1
        #     a_c = a + alpha * p
        #     s = (a_c - a).reshape((k, 1))
        #     grad_c = approx_fprime(a_c, f, np.repeat(1e-8, a.size, axis=0), x, m)
        #     y = (grad_c - grad).reshape((k, 1))
        #     rho = 1 / (y.transpose().dot(s))
        #     h = h.dot(np.eye(k) - rho * s.dot(y.transpose()))
        #     h = (np.eye(k) - rho * s.dot(y.transpose())).dot(h) + rho * y.dot(y.transpose())
        #
        #     grad, a = grad_c, a_c
        #     i += 1

        if graph:
            x = x.sort_values()
            npdf = np.vectorize(norm.pdf)
            # real = npdf(x, loc=-6, scale=2.15)
            appr = self.a_pdf(x, a, moments_calculated=True, moments=m)
            df = pd.DataFrame({'x': x, 'Approximation': appr})
            # plt.plot(x, norm.pdf(x, loc=-6, scale=2.15), color='purple', marker='.')
            # plt.plot(x, self.a_pdf(x, a, moments_calculated=True, moments=m), color='plum', marker='.')
            # plt.show()
            fig = px.line(df, x='x', y='Approximation')
            fig.show()
        return a


# sample = pd.Series(np.random.normal(loc=-6, scale=2.15, size=5000))
# test = Optimize(alpha=sample.mean(), beta=math.sqrt(3 * sample.var()) / math.pi)
# param = test.gd(test.log_likelihood, np.array([1, 0, 0, 0, 0, 0]), sample, truncated=True, minimize=True)
# test1 = Optimize()
# print(test1.moment(2))
