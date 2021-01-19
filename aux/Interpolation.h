#ifndef CW_INTERPOLATION_H
#define CW_INTERPOLATION_H
#include <cmath>
#include <Eigen/Core>

using Eigen::MatrixXd, Eigen::VectorXd;

double cubic_minimizer(double alpha_lo, double f_lo, double df_lo,
        double alpha_hi, double f_hi,
        double alpha_i, double f_i) {
    double alpha_0 = alpha_i - alpha_lo;
    double alpha_1 = alpha_hi - alpha_lo;

    double denominator = alpha_0 * alpha_0 * alpha_1 * alpha_1 * (alpha_1 - alpha_0);

    MatrixXd alpha(2, 2);
    alpha << alpha_0 * alpha_0,
    -alpha_1 * alpha_1,
    -alpha_0 * alpha_0 * alpha_0,
    alpha_1 * alpha_1 * alpha_1;

    VectorXd phi(2);
    phi << f_hi - f_lo - df_lo * alpha_1,
    f_i - f_lo - df_lo * alpha_0;

    VectorXd AB(2);
    AB = alpha * phi;
    double a = AB[0] / denominator;
    double b = AB[1] / denominator;

    double alpha_j = alpha_lo + (-b * sqrt(b * b - 3 * a * df_lo)) / (3 * a);
    return alpha_j;
}

double quadratic_minimizer(double alpha_lo, double f_lo, double df_lo,
                           double alpha_hi, double f_hi) {
    double alpha_j = df_lo * (alpha_hi - alpha_lo) * (alpha_hi - alpha_lo);
    alpha_j /= 2 * (f_hi - f_lo + (alpha_hi - alpha_lo) * df_lo);
    alpha_j *= -1;
    alpha_j += alpha_lo;
    return alpha_j;
}

#endif //CW_INTERPOLATION_H
