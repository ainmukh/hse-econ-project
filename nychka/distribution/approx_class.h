#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Core>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_zeta.h>
#include <gsl/gsl_sf_fermi_dirac.h>
#include <vector>
//#include <iostream>

using Eigen::Matrix;
using Eigen::VectorXd;

void li(size_t k, double x, Matrix<double, -1, 1> &ans);

template<typename T>
class Distribution {
private:
    T loc;
    T scale;
    Matrix<T, -1, 1> param;
    size_t k;
    //
    Matrix<T, -1, 1> C;
    Matrix<T, -1, 1> C_inf;
    Matrix<T, -1, 1> Gamma;
    Matrix<T, -1, -1> C_li;
    Matrix<T, -1, -1> Col;
    Matrix<T, -1, -1> Col_arg;
    Matrix<T, -1, -1> Alpha;
    Matrix<T, -1, 1> M;
    Matrix<T, -1, 1> C_y;
    std::vector<T> y_abs_pow;
    Matrix<T, -1, 1> X_pow;
    Matrix<T, -1, 1> Li;
    Matrix<T, -1, 1> I;
    Matrix<T, -1, 1> I_u;
    Matrix<T, -1, -1> I_t;
public:
    Distribution(T loc_, T scale_, Matrix<T, -1, 1> param_)
            : loc(loc_), scale(scale_), param(std::move(param_)), k(param.size()),
              C_y(Matrix<T, -1, 1>::Ones(2 * k - 2)), y_abs_pow(2 * k - 1), X_pow(Matrix<T, -1, 1>(2 * k - 2)),
              Li(Matrix<T, -1, 1>(2 * k - 2)), I_u(Matrix<T, -1, 1>(2 * k - 1)) {
        // Vector of i * Gamma(i) * (1 - 2^{1 - i}) * zeta(i)
        C = Matrix<T, -1, 1>(2 * k - 2);
        C[0] = M_LN2;

        {
            double curr = 0.5;
            for (size_t i = 2; i <= 2 * k - 2; i++) {
                C[i - 1] = i * gsl_sf_fact(i - 1) * (1 - curr) * gsl_sf_zeta_int(i);
                curr /= 2;
            }
        }

        C_inf = Matrix<T, -1, 1>::Ones(2 * k - 2);
        for (size_t i = 0; i < 2 * k - 2; i += 2) {
            C_inf[i] = -1;
        }

        // Vector of i * Gamma(i)
        Gamma = Matrix<T, -1, 1>(2 * k - 2);

        C_li = Matrix<T, -1, -1>::Zero(2 * k - 2, 2 * k - 2);
        for (size_t i = 0; i < 2 * k - 2; i++) {
            Gamma[i] = gsl_sf_fact(i + 1);
            if (i != 0) {
                C_li(i, 0) = 1 / Gamma[i - 1];
            } else {
                C_li(i, 0) = 1;
            }
        }

        // Matrix of collocation coefficients
        Col = Matrix<T, -1, -1>::Zero(2 * k - 1, 2 * k - 1);
        for (size_t i = 0; i < 2 * k - 1; i++) {
            for (size_t j = 0; j <= i; j++) {
                Col(i, j) = gsl_sf_choose(i, j);
            }
        }

        M = Matrix<T, -1, 1>(2 * k - 1);
        M[0] = 1;
        for (size_t i = 1; i < 2 * k - 1; i++) {
            M[i] = i % 2 == 1 ? 0 : 2 * C[i - 1];
        }

        precalc();
    }

    void precalc() {
        // Matrix of collocation coefficients
        std::vector<T> locs(2 * k - 1), scales(2 * k - 1);
        locs[0] = 1;
        scales[0] = 1;
        for (size_t i = 1; i < 2 * k - 1; i++) {
            locs[i] = locs[i - 1] * loc;
            scales[i] = scales[i - 1] * scale;
        }

        Col_arg = Col;
        for (size_t i = 0; i < 2 * k - 1; i++) {
            for (size_t j = 0; j <= i; j++) {
                Col_arg(i, j) *= locs[i - j] * scales[j];
            }
        }

        // Matrix of shifted alpha parameter vector
        Alpha = Matrix<T, -1, -1>::Zero(k, 2 * k - 1);
        for (size_t i = 0; i < k; i++) {
            for (size_t j = i; j < i + k; j++) {
                Alpha(i, j) = param[j - i];
            }
        }
        Alpha = param.transpose() * Alpha;
    }

    void reset_param(T loc_, T scale_, Eigen::Matrix<T, -1, 1> &param_) {
        loc = loc_;
        scale = scale_;
        param = param_;
        precalc();
    }

    double cdf(T x) {
//        std::cout << "x = " << x << std::endl;
        T y = (x - loc) / scale;
        T y_abs = abs(y);

        if (y < 0) {
            for (size_t i = 1; i < 2 * k - 2; i += 2) {
                C_y[i] = -1;
            }
        } else {
            for (size_t i = 1; i < 2 * k - 2; i += 2) {
                C_y[i] = 1;
            }
        }

        // Vector of pow(arg, i)
        y_abs_pow[0] = 1;
        for (size_t i = 1; i < 2 * k - 1; i++) {
            y_abs_pow[i] = y_abs_pow[i - 1] * y_abs;
        }

        for (size_t i = 0; i < 2 * k - 2; i++) {
            X_pow[i] = y_abs_pow[i + 1]; // (y_abs, i + 1);
        }
        X_pow = (0.5 * tanh(y_abs / 2) - 0.5) * X_pow;

        // Polylogarithms vector

//        for (size_t i = 0; i < 2 * k - 2; i++) {
//            if (y_abs >= 690) Li[i] = 0;
//            else Li[i] = -gsl_sf_fermi_dirac_int(i, -y_abs);
//        }
        double z = -std::exp(-y_abs);
        li(2 * k - 2, z, Li);
//        std::cout << "li = " << Li.transpose() << std::endl;

        // Polylogarithm coefficients
        Matrix<T, -1, -1> C_li_arg = C_li;
        for (size_t i = 0; i < 2 * k - 2; i++) {
            C_li_arg(i, 0) *= y_abs_pow[i];  // pow(y_abs, i);
            for (size_t j = 1; j <= i; j++) {
                C_li_arg(i, j) = C_li_arg(i - 1, j - 1);
            }
        }

        C_li_arg = C_li_arg * Li;
        C_li_arg = Gamma.array() * C_li_arg.array();

        I = (C_inf + C_y).array() * C.array() + C_y.array() * X_pow.array() + C_y.array() * C_li_arg.array();

        // from -infty to 0 + 0.5 * tanh
        T C_y0 = y >= 0 ? 1 : -1;
        I_u << 0.5 + C_y0 * 0.5 * tanh(y_abs / 2), I;

        // Vector I^t
        I_t = Col_arg * I_u;
        T fx = (Alpha * I_t)(0);
        T C_1 = 1 / (Alpha * Col_arg * M).sum();
//        std::cout << "fx = " << C_1 * fx << std::endl;
        return C_1 * fx;
    }
};
