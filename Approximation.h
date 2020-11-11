#ifndef CW_APPROXIMATION_H
#define CW_APPROXIMATION_H

#include <Eigen/Core>
#include <cmath>
#include <gsl_sf_fermi_dirac.h>  // Polylogarithm
#include <gsl_sf_gamma.h>
#include <gsl_sf_zeta.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Array;

class CDF {
private:
    double alpha;
    double beta;
    size_t n;  // k + 1
    VectorXd param;  // alphas vector

public:
    CDF(const VectorXd& param_) : param(param_), n(param_.size())
    {
        alpha = 0.0;
        beta = 1.0;
    }

    void reset_param(const VectorXd& param_) {
        param = param_;
    }

    std::pair<double, VectorXd> operator() (const VectorXd& x, const VectorXd& w) {
        size_t w_size = w.size();

        // F(x)
        // C
        // Vector of i * Gamma(i) * (1 - 2^{1 - i}) * zeta(i)
        VectorXd C = VectorXd::Zero(2 * n - 2);
        C[0] = M_LN2;
        for (size_t i = 2; i <= 2 * n - 2; i++) {
            C[i - 1] = i * gsl_sf_gamma(i) * (1 - pow(2, 1 - static_cast<double>(i))) * gsl_sf_zeta_int(i);
        }

        // Vector of int from 0 to arg sign
        VectorXd C_y = VectorXd::Ones(2 * n - 2);
        if ((x.dot(w) - alpha) / beta < 0) {
            for (size_t i = 0; i < 2 * n - 2; i += 2) {
                C_y[i] = -1;
            }
        }

        // Vector of int from -infinity to 0 sign
        VectorXd C_inf = VectorXd::Ones(2 * n - 2);
        for (size_t i = 1; i < 2 * n - 2; i += 2) {
            C_inf[i] = - 1;
        }
        // C

        // The 2nd term
        // Vector of pow(arg, i)
        VectorXd X_pow = VectorXd::Zero(2 * n - 2);
        for (size_t i = 0; i < 2 * n - 2; i++) {
            X_pow[i] = pow(abs((x.dot(w) - alpha) / beta), i + 1);
        }
        X_pow = (0.5 * tanh(abs((x.dot(w) - alpha) / beta) / 2) - 0.5) * X_pow;
        // The 2nd term

        // The 3d term
        // Polylogarithms vector
        VectorXd Li = VectorXd::Zero(2 * n - 2);
        for (size_t i = 0; i < 2 * n - 2; i++) {
            Li[i] = -gsl_sf_fermi_dirac_int(i, -abs((x.dot(w) - alpha) / beta));
        }

        // Vector of i * Gamma(i)
        VectorXd Gamma = VectorXd::Zero(2 * n - 2);
        // Polylogarithm coefficients
        MatrixXd C_li = MatrixXd::Zero(2 * n - 2, 2 * n - 2);
        for (size_t i = 0; i < 2 * n - 2; i++) {
            Gamma[i] = (static_cast<double>(i) + 1) * gsl_sf_gamma(static_cast<double>(i) + 1);

            C_li(i, 0) = pow(abs((x.dot(w) - alpha) / beta), i)
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
        I_u << 0.5 + 0.5 * tanh(abs((x.dot(w) - alpha) / beta) / 2), I;
        // Vector I^u

        // Vector I^t
        VectorXd I_t = VectorXd::Zero(2 * n - 1);
        // Matrix of collocation coefficients
        MatrixXd Col = MatrixXd::Zero(2 * n - 1, 2 * n - 1);
        for (size_t i = 0; i < 2 * n - 1; i++) {
            for (size_t j = 0; j <= i; j++) {
                Col(i, j) = gsl_sf_gamma(static_cast<double>(i) + 1)
                            / (gsl_sf_gamma(static_cast<double>(j) + 1 + 1)
                               * gsl_sf_gamma(static_cast<double>(i) - static_cast<double>(j) + 1));
                Col(i, j) *= pow(alpha, i - j) * pow(beta, j);
            }
        }
        // Vector I^t

        I_t = Col * I_u;

        // Matrix of shifted alpha parameter vector
        MatrixXd Alpha = MatrixXd::Zero(n, 2 * n - 1);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i; j < i + n; j++) {
                Alpha(i, j) = param[j - i];
            }
        }
        double fx = (param.transpose() * Alpha * I_t)[0];
        // Matrix of shifted alpha parameter vector

        // Moments
        // Vector of full moments
        VectorXd M = VectorXd::Zero(2 * n - 1);
        M[0] = 1;
        for (size_t i = 1; i < 2 * n - 1; i++) {
            M[i] = i % 2 == 1 ? 0 : 2 * C[i - 1];
        }
        double C_1 = 1 / (Col * M).sum();
        // Moments
        // F(x)

        // GRADIENT
        // d/d alpha
        VectorXd A = VectorXd::Zero(n);
        VectorXd E = VectorXd::Zero(n);

        // Vector of even I^t
        VectorXd I_ta = VectorXd::Zero(n);
        // Vector of even moments
        VectorXd M_e = VectorXd::Zero(n);
        for (size_t i = 0; i < n; i++) {
            I_ta[i] = I_t[2 * i];
            M_e[i] = M[2 * i];
        }

        A = param.array() * I_ta.array();
        A += Alpha * I_t;

        E = param.array() * M_e.array();
        E += Alpha * M;

        VectorXd dfa = VectorXd::Zero(n);
        dfa = C_1 * A + C_1 * C_1 * E * (param.transpose() * Alpha * I_t);
        // d/d alpha

        // d/d w
        VectorXd wx = VectorXd::Zero(2 * n - 1);
        for (size_t i = 0; i < 2 * n - 1; i++) {
            wx[i] = pow(x.dot(w), i);
            if (i % 2 == 0 && x.dot(w) < 0) {
                wx[i] *= -1;
            }
        }

        VectorXd dfw = VectorXd::Zero(w_size);
        dfw = x * C_1 * param.transpose() * Alpha * wx;
        dfw = dfw * exp(-(x.dot(w) - alpha) / beta)
              / (beta * pow(1 + exp(-(x.dot(w) - alpha) / beta), 2));
        // d/d w

        VectorXd df = VectorXd::Zero(n + w_size);
        df << dfa, dfw;
        // GRADIENT

        return {C_1 * fx, df};
    }
};

#endif //CW_APPROXIMATION_H
