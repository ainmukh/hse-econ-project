//
// Created by Ainur Mukhambetova on 11.12.2020.
//

#ifndef CW_APPROXIMATION_H
#define CW_APPROXIMATION_H

#include <Eigen/Core>
#include <cmath>
#include <gsl_sf_fermi_dirac.h>  // Polylogarithm
#include <gsl_sf_gamma.h>
#include <gsl_sf_zeta.h>
#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <tuple>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Array;

std::tuple<double, double*, size_t> CDF_func(
        const double alpha,
        const double beta,
        const double* alphas_,
        const double* theta_,
        const double* x_,
        size_t alphas_size, size_t theta_size, size_t x_size
) {
    // переход от вектора плюсов к eigen
    VectorXd alphas(alphas_size);
    VectorXd theta(theta_size);
    VectorXd x(x_size);

    for (size_t i = 0; i < fmax(alphas.size(), theta.size()); i++) {
        if (i < alphas_size) alphas[i] = alphas_[i];
        if (i < theta_size) {
            theta[i] = theta_[i];
            x[i] = x_[i];
        }
    }

    // F(x)
    size_t n = alphas_size;
    // C
    // Vector of i * Gamma(i) * (1 - 2^{1 - i}) * zeta(i)
    VectorXd C = VectorXd::Zero(2 * n - 2);
    C[0] = M_LN2;
    for (size_t i = 2; i <= 2 * n - 2; i++) {
        C[i - 1] = i * gsl_sf_gamma(i) * (1 - pow(2, 1 - static_cast<double>(i))) * gsl_sf_zeta_int(i);
    }

    // Vector of int from 0 to arg sign
    VectorXd C_y = VectorXd::Ones(2 * n - 2);
    if ((x.dot(theta) - alpha) / beta < 0) {
        for (size_t i = 1; i < 2 * n - 2; i += 2) {
            C_y[i] = -1;
        }
    }

    // Vector of int from -infinity to 0 sign
    VectorXd C_inf = VectorXd::Ones(2 * n - 2);
    for (size_t i = 0; i < 2 * n - 2; i += 2) {
        C_inf[i] = - 1;
    }
    // C

    // The 2nd term
    // Vector of pow(arg, i)
    VectorXd X_pow = VectorXd::Zero(2 * n - 2);
    for (size_t i = 0; i < 2 * n - 2; i++) {
        X_pow[i] = pow(abs((x.dot(theta) - alpha) / beta), i + 1);
    }
    X_pow = (0.5 * tanh(abs((x.dot(theta) - alpha) / beta) / 2) - 0.5) * X_pow;
    // The 2nd term

    // The 3d term
    // Polylogarithms vector
    VectorXd Li = VectorXd::Zero(2 * n - 2);
    for (size_t i = 0; i < 2 * n - 2; i++) {
        if (abs((x.dot(theta) - alpha) / beta) >= 690) Li[i] = 0;
        else Li[i] = -gsl_sf_fermi_dirac_int(i, -abs((x.dot(theta) - alpha) / beta));
    }

    // Vector of i * Gamma(i)
    VectorXd Gamma = VectorXd::Zero(2 * n - 2);
    // Polylogarithm coefficients
    MatrixXd C_li = MatrixXd::Zero(2 * n - 2, 2 * n - 2);
    for (size_t i = 0; i < 2 * n - 2; i++) {
        Gamma[i] = (static_cast<double>(i) + 1) * gsl_sf_gamma(static_cast<double>(i) + 1);

        C_li(i, 0) = pow(abs((x.dot(theta) - alpha) / beta), i)
                     / gsl_sf_gamma(static_cast<double>(i) + 1);
        for (size_t j = 1; j <= i; j++) {
            C_li(i, j) = C_li(i - 1, j - 1);
        }
    }

    C_li = C_li * Li;
    C_li = Gamma.array() * C_li.array();
    // The 3d term

    // Vector I^u
    VectorXd I = (C_inf + C_y).array() * C.array() + C_y.array() * X_pow.array() + C_y.array() * C_li.array();
    VectorXd I_u = VectorXd::Zero(2 * n - 1);
    // from -infty to 0 + 0.5 * tanh
    double C_y0 = (x.dot(theta) - alpha) / beta >= 0 ? 1 : -1;
    I_u << 0.5 + C_y0 * 0.5 * tanh(abs((x.dot(theta) - alpha) / beta) / 2), I;
    // Vector I^u

    // Vector I^t
    VectorXd I_t = VectorXd::Zero(2 * n - 1);
    // Matrix of collocation coefficients
    MatrixXd Col = MatrixXd::Zero(2 * n - 1, 2 * n - 1);
    for (size_t i = 0; i < 2 * n - 1; i++) {
        for (size_t j = 0; j <= i; j++) {
            Col(i, j) = gsl_sf_gamma(static_cast<double>(i) + 1) /
                        ((gsl_sf_gamma(static_cast<double>(j) + 1)
                          * gsl_sf_gamma(static_cast<double>(i) - static_cast<double>(j) + 1)));

            Col(i, j) *= pow(alpha, i - j) * pow(beta, j);
        }
    }
    // Vector I^t
    I_t = Col * I_u;

    // Matrix of shifted alpha parameter vector
    MatrixXd Alpha = MatrixXd::Zero(n, 2 * n - 1);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < i + n; j++) {
            Alpha(i, j) = alphas[j - i];
        }
    }
    double fx = (alphas.transpose() * Alpha * I_t)[0];
    // Matrix of shifted alpha parameter vector

    // Moments
    // Vector of full moments
    VectorXd M = VectorXd::Zero(2 * n - 1);
    M[0] = 1;
    for (size_t i = 1; i < 2 * n - 1; i++) {
        M[i] = i % 2 == 1 ? 0 : 2 * C[i - 1];
    }
    double C_1 = 1 / (alphas.transpose() * Alpha * Col * M).sum();
    // Moments
    // F(x)

    // GRADIENT
    // d/d alpha
    VectorXd A = VectorXd::Zero(n);
    VectorXd B = VectorXd::Zero(n);

    A = 2 * Alpha * I_t;
    B = 2 * Alpha * M;

    VectorXd dfa = VectorXd::Zero(n);
    dfa = C_1 * A - C_1 * C_1 * B * (alphas.transpose() * Alpha * I_t);
    // d/d alpha

    // d/d w
    VectorXd wx = VectorXd::Zero(2 * n - 1);
    for (size_t i = 0; i < 2 * n - 1; i++) {
        wx[i] = pow(x.dot(theta), i);
        if (i % 2 == 0 && x.dot(theta) < 0) {
            wx[i] *= -1;
        }
    }

    VectorXd dfw = VectorXd::Zero(theta.size());
    dfw = x * C_1 * alphas.transpose() * Alpha * wx;
    dfw = dfw * exp(-(x.dot(theta) - alpha) / beta)
          / (beta * pow(1 + exp(-(x.dot(theta) - alpha) / beta), 2));
    // d/d w

    VectorXd df = VectorXd::Zero(n + theta.size());
    df << dfa, dfw;
    // GRADIENT

    double* df_ = new double[df.size()];
    if (df_ == NULL) {
        throw std::runtime_error("failed to allocate result array");
    }
    std::copy(df.data(), df.data() + df.size(), df_);
    return {C_1 * fx, df_, df.size()};
}

#endif //CW_APPROXIMATION_H
