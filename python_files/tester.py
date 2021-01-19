# --------
# Для Айнуры
# --------

import sys
import math
import statistics
import numpy as np
import scipy
from scipy import stats
import cdf_module_np


# Рассчитываем, приблизительно, момент усеченного
# логистического распределения при помощи ЗБЧ
def logistic_tr_moment(k,                                # рассчитываемый момент
                       loc=0, scale=1,                   # параметры логистического распределения
                       tr_lower=-math.inf,               # нижняя граница усечения
                       tr_upper=math.inf,                # верхняя граница усечения
                       t=None,                           # выборка из усеченного, в точках tr_lower и tr_upper,
                                                         # логистического распределения с параметрами loc и scale
                       n=1000000):                       # объем выборки t (чем больше, тем точнее рассчеты)
    if t is None:                                        # если выборка по умолчанию не задана, то:
        t = scipy.stats.logistic.rvs(loc, scale, n)      # формируем выборку из распределения с параметрами loc и scale
        t = t[(t >= tr_lower) & (t <= tr_upper)]         # осуществляем усечение в точках tr_lower и tr_upper
    return np.mean(t ** k)                               # аппроксимируем истинное значение момента при помощи
                                                         # выборочного момента


# Расчет функции распределения Галланта и Нички с
# логистическим ядром
def F_GN(x,                                               # точка, в которой осуществляется расчет
         loc,                                             # параметры логистического
         scale,                                           # распределения
         alpha, n=1000000):                               # параметры полинома
    n_alpha = alpha.size                                  # число коэффициентов в полиноме (на 1 больше, чем степень)
    moments = np.empty(2 * n_alpha)                       # массивы, в которые сохраняются обычные и
    moments_tr = np.empty(2 * n_alpha)                    # усеченные моменты логистического распределения
    F_logistic = scipy.stats.logistic.cdf(x, loc, scale)  # функция распределения обычного логистического
                                                          # распределения в соответствующей точке
    F_nom = 0                                             # числитель выражения, используемого для расчетов
    F_den = 0                                             # знаменатель выражения, используемого для расчетов
    for i in range(0, 2 * n_alpha):
        moments[i] = scipy.stats.logistic.moment(         # считаем обычные моменты
            n=i, loc=loc, scale=scale)
        moments_tr[i] = logistic_tr_moment(               # считаем усеченные моменты (тут нужна быстрая функция)
            k=i, loc=loc,
            scale=scale, tr_upper=x, n=n)
    for i in range(0, n_alpha):
        for j in range(0, n_alpha):
            alpha_prod = alpha[i] * alpha[j]              # осуществляем основные расчеты
            F_nom += alpha_prod * moments_tr[i + j]
            F_den += alpha_prod * moments[i + j]
    return F_logistic * F_nom / F_den                     # аккумулируем результат


# Пример расчета
# F_value = F_GN(x=1.6,
#                loc=2.2, scale=3.5,
#                alpha=np.array([1, 0.3, -0.5]), n=1000000)

alphas = np.array([1, 0.3, -0.5, 0.07], dtype=np.float64)
theta = np.array([1], dtype=np.float64)
x = np.array([1.6], dtype=np.float64)

Fx, dfx = cdf_module_np.CDF(2.2, 3.5, alphas, theta, x)

m = 100000

for i in range(3):
    m *= 10
    F_value = F_GN(x=1.6,
                   loc=2.2, scale=3.5,
                   alpha=np.array([1, 0.3, -0.5]), n=m)
    abs = round(np.abs(Fx - F_value), 5)
    rel = round(np.abs(Fx - F_value) * 100 / F_value, 3)
    print(f'n = {m}, abs = {abs}, rel = {rel}%')

# Айнура, проверь, пожалуйста, сходятся ли расчеты
# моментов и функции распределения по твоим формулами
# с теми, что получаются по этим функциям.
# Если у тебя все верно, то погрешность должна быть не
# больше 5%-10%. Чем больше симулируемая выборка t,
# тем дольше расчеты, но тем меньше погрешность.
