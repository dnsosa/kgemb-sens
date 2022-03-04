# -*- coding: utf-8 -*-

"""Methods for calculating metrics about network and test edge."""

import numpy as np
import powerlaw

from scipy import log
from scipy.special import zeta
from scipy.optimize import bisect, curve_fit
from scipy import sqrt


def calc_scale_free_stats(degree_dict, sf=4):
    in_data = list(degree_dict.values())
    in_data_x = sorted(set(degree_dict.values()))
    in_data_y = [list(degree_dict.values()).count(x) for x in in_data_x]
    n = len(in_data)

    # Least squares fit
    def power_law_1(x, b):
        return np.power(x, b)
    def power_law_2(x, a, b):
        return a * np.power(x, b)

    pars1, cov1 = curve_fit(f=power_law_1, xdata=in_data_x, ydata=in_data_y)
    lsf_b1 = round(pars1[0], sf)
    lsf_b1_std = round(cov1[0, 0]**.5, sf)
    pars2, cov2 = curve_fit(f=power_law_2, xdata=in_data_x, ydata=in_data_y)
    lsf_a2, lsf_b2 = round(pars2[0], sf), round(pars2[1], sf)
    lsf_a2_std, lsf_b2_std = round(cov2[0, 0]**.5, sf), round(cov2[1, 1]**.5, sf)
    lsf_stats = ((lsf_b1, lsf_b1_std), (lsf_a2, lsf_a2_std, lsf_b2, lsf_b2_std))

    # Discrete MLE fit for alpha, based on what I read online
    # (https://www.johndcook.com/blog/2015/11/24/estimating-the-exponent-of-discrete-power-law-data/)
    xmin = 1

    def log_zeta(x):
        return log(zeta(x, 1))

    def log_deriv_zeta(x):
        h = 1e-5
        return (log_zeta(x + h) - log_zeta(x - h)) / (2 * h)

    t = -sum(log(np.array(in_data) / xmin)) / n

    def objective(x):
        return log_deriv_zeta(x) - t

    def zeta_prime(x, xmin=1):
        h = 1e-5
        return (zeta(x + h, xmin) - zeta(x - h, xmin)) / (2 * h)

    def zeta_double_prime(x, xmin=1):
        h = 1e-5
        return (zeta(x + h, xmin) - 2 * zeta(x, xmin) + zeta(x - h, xmin)) / h ** 2

    def sigma(n, alpha_hat, xmin=1):
        z = zeta(alpha_hat, xmin)
        temp = zeta_double_prime(alpha_hat, xmin) / z
        temp -= (zeta_prime(alpha_hat, xmin) / z) ** 2
        return 1 / sqrt(n * temp)

    a, b = 1.01, 10
    disc_mle_alpha_hat = round(bisect(objective, a, b, xtol=1e-6), sf)
    # print(alpha_hat)
    disc_mle_sig = round(sigma(n, disc_mle_alpha_hat)**.5, sf)
    # print(f"MLE fit: {alpha_hat} (95% CI: [{alpha_hat - 2 * sig},{alpha_hat + 2 * sig}])")
    disc_mle_stats = (disc_mle_alpha_hat, disc_mle_sig)

    # Power law fit using the powerlaw package
    results = powerlaw.Fit(in_data)
    # print(f"MLE fit (powerlaw package) - alpha: {round(results.power_law.alpha, sf)}")
    # print(f"MLE fit (powerlaw package) - xmin: {round(results.power_law.xmin, sf)}")

    pl_alpha = round(results.power_law.alpha, sf)
    pl_xmin = round(results.power_law.xmin, sf)
    pl_R, pl_p = results.distribution_compare('power_law', 'lognormal')

    powerlaw_stats = (pl_alpha, pl_xmin, pl_R, pl_p)

    return lsf_stats, disc_mle_stats, powerlaw_stats
