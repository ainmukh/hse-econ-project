//
// https://github.com/scipy/scipy/blob/master/scipy/optimize/minpack2/dcsrch.f
// https://github.com/scipy/scipy/blob/master/scipy/optimize/minpack2/dcstep.f
//

#ifndef CW_LINESEARCH_H
#define CW_LINESEARCH_H

#include <utility>
#include <iostream>
#include "Param.h"
#include "Interpolation.h"

using Eigen::VectorXd;

template <typename F>
class LineSearch {
private:
    F& f;
    const VectorXd& x;
    const VectorXd& p;
    size_t n;
    BFGSParam param;

public:
    LineSearch(F& f_, const VectorXd&  x_, const VectorXd&  p_, BFGSParam param_)
            : f(f_), x(x_), p(p_), n(x.size()), param(param_) {}

    // Numerical Optimization
    // Nocedal, Wright page 60
    // Algorithm 3.5
    void bracketing(double f_0_prev) {
        // alpha_{i - 1} | alpha_p – previous step length
        // alpha_{i}     | alpha_c – current step length
        VectorXd df_0 = VectorXd::Zero(n);
        double f_0 = f(x, df_0);

        // LINE SEARCH 1
        double alpha_i;
        alpha_i = fmin(1.0, 1.01 * 2 * (f_0 - f_0_prev) / df_0.dot(p));
        if (alpha_i < 0) {
            alpha_i = 1.0;
        }

        VectorXd df_i = VectorXd::Zero(n);
        double f_i = f(x + alpha_i * p, df_i);

        double alpha_lo = 0;
        double f_lo = f_0;
        VectorXd df_lo = df_0;
        double alpha_hi = 0;
        double f_hi = f_0;
        VectorXd df_hi = df_0;

        size_t i = 0;
        size_t stage = 1;

        bool bracketed = false;

        double width = param.alpha_max - param.alpha_min;
        double width_1 = width * 2;

        double a_min = 0;
        double a_max = alpha_i + param.xtrapu * alpha_i;

        // Check the input arguments for errors.
        if (alpha_i < param.alpha_min || alpha_i > param.alpha_max)
            throw std::invalid_argument("'alpha_i' must satisfy alpha_min < alpha < alpha_max");
        if (df_0.dot(p) >= 0)
            throw std::invalid_argument("initial gradient must be negative");


        for(;;)
        {
            if (i >= param.max_line_search) break;

            double alpha_j = search(alpha_i, f_i, df_i,
                    alpha_lo, f_lo, df_lo,
                    alpha_hi, f_hi, df_hi,
                    f_0, df_0,
                    stage, bracketed, width, width_1,
                    a_min, a_max);

            switch(param.result) {
                case LSResult::CONVERGED:
                    param.alpha = alpha_j;
                    return;
                case LSResult::FG:
                    alpha_i = alpha_j;
                    df_i = VectorXd::Zero(n);
                    f_i = f(x + alpha_i * p, df_i);
                    break;
                default:
                    i = param.max_line_search;
            }
            i++;
        }

        // LINE SEARCH 2
        double alpha_c = fmin(1.0, 1.01 * 2 * (f_0 - f_0_prev) / df_0.dot(p));
        if (alpha_c < 0) {
            alpha_c = 1.0;
        }

        double alpha_p = 0;
        double f_prev = f_0;
        VectorXd df_prev = df_0;

        param.result = LSResult::FG;

        i = 0;

        for (;;)
        {
            if (i >= param.max_line_search) break;

            if (alpha_c == 0 || alpha_p == a_max) {
                param.result = LSResult::WARNING;
                return;
            }

            VectorXd df_curr = VectorXd::Zero(n);
            double f_curr = f(x + alpha_i * p, df_curr);

            // Condition i and ii
            if (f_curr > f_0 + param.c_1 * alpha_c * df_0.dot(p) || f_curr >= f_prev && i > 1)
            {
                selection(alpha_p, f_prev, df_prev,
                        alpha_c, f_curr, df_curr,
                        f_0, df_0);
                break;
            }

            // Termination condition and
            // Condition iii
            if (abs(df_curr.dot(p)) <= -param.c_2 * df_0.dot(p)) {
                param.alpha = alpha_c;
                param.result = LSResult::CONVERGED;
                break;
            } else if (df_curr.dot(p) >= 0) {
                selection(alpha_c, f_curr, df_curr,
                        alpha_p, f_prev, df_prev,
                        f_0, df_0);
                break;
            }

            // Extrapolation to find the next trial value
            // To implement this step we can simply set alpha_{i + 1} to some constant multiple of alpha_{i}.
            // It is important that the successive steps increase quickly enough
            // to reach the upper limit in a finite number of iterations
            double tmp = alpha_c;
            alpha_c = fmin(2 * alpha_c,  // I used 1.1 instead of 2
                           param.alpha_max);  // must be amax
            alpha_p = tmp;

            f_prev = f_curr;
            df_prev = df_curr;

            i++;
        }

        switch(param.result) {
            case LSResult::CONVERGED:
                break;
            default:
                std::cerr << "WARNING: Line Search has not converged" << std::endl;
        }
    }

    // Numerical Optimization
    // Nocedal, Wright page 61
    // Algorithm 3.6
    void selection(double alpha_lo, double f_lo, VectorXd df_lo,
            double alpha_hi, double f_hi, VectorXd df_hi,
            double f_0, const VectorXd& df_0) {
        size_t i = 0;

        double alpha_i = 0;
        double f_i = f_0;
        VectorXd df_i = df_0;

        for (;;)
        {
            if (i >= param.max_line_search) break;

            double alpha_j;
            double a = fmin(alpha_lo, alpha_hi);
            double b = fmax(alpha_lo, alpha_hi);
            double cubic_check = 0.2 * (alpha_hi - alpha_lo);
            if (i > 0) {
                alpha_j = cubic_minimizer(alpha_lo, f_lo, df_lo.dot(p),
                                          alpha_hi, f_hi,
                                          alpha_i, f_i);
            }
            if (i == 0 || i > 0 && (alpha_j > b - cubic_check || alpha_j < a + cubic_check)) {
                double quadratic_check = 0.1 * (alpha_hi - alpha_lo);
                alpha_j = quadratic_minimizer(alpha_lo, f_lo, df_lo.dot(p),
                                              alpha_hi, f_hi);
                if (alpha_j > b - quadratic_check || alpha_j < a + quadratic_check) {
                    alpha_j = alpha_lo + (alpha_hi - alpha_lo) / 2;
                }
            }

            VectorXd df_j = VectorXd::Zero(n);
            double f_j = f(x + alpha_j * p, df_j);

            if (f_j > f_0 + param.c_1 * alpha_j * df_0.dot(p) || f_j >= f_lo) {
                alpha_i = alpha_hi;
                f_i = f_hi;
                df_i = df_hi;

                alpha_hi = alpha_j;
                f_hi = f_j;
                df_hi = df_j;
            } else {
                if (abs(df_j.dot(p)) <= -param.c_2 * df_0.dot(p)) {
                    // Converged
                    param.result = LSResult::CONVERGED;
                    param.alpha = alpha_j;
                    return;
                }
                if (abs(df_j.dot(p)) * (alpha_hi - alpha_lo) >= 0) {
                    alpha_i = alpha_hi;
                    f_i = f_hi;
                    df_i = df_hi;

                    alpha_hi = alpha_lo;
                    f_hi = f_lo;
                    df_hi = df_lo;
                } else {
                    alpha_i = alpha_lo;
                    f_i = f_lo;
                    df_i = df_lo;
                }
                alpha_lo = alpha_j;
                f_lo = f_j;
                df_lo = df_j;
            }
            i++;
        }
    }

    // This subroutine computes a safeguarded step for a search procedure
    // and updates an interval that contains a step
    // that satisfies a sufficient decrease and a curvature condition.
    // It was a cubic interpolation once
    double step(double& alpha_lo, double& f_lo, VectorXd& df_lo,
            double& alpha_hi, double& f_hi, VectorXd& df_hi,
            double alpha_i, double f_i, const VectorXd& df_i,
            bool& bracketed,
            double a_min, double a_max) {

        double alpha_j;

        if (f_i > f_lo) {
            double z = 3 * (f_lo - f_i) / (alpha_i - alpha_lo) + df_lo.dot(p) + df_i.dot(p);
            double w = sqrt(z * z - df_lo.dot(p) * df_i.dot(p));

            if (alpha_i < alpha_lo) {
                w *= -1;
            }

            double alpha_c = (w - df_lo.dot(p) + z) * (alpha_i - alpha_lo);
            alpha_c /= (df_i.dot(p) - df_lo.dot(p) + 2 * w);
            alpha_c += alpha_lo;

            double alpha_q = (alpha_i - alpha_lo) * (alpha_i - alpha_lo) * df_lo.dot(p);
            alpha_q /= 2 * (f_lo - f_i + (alpha_i - alpha_lo) * df_lo.dot(p));
            alpha_q += alpha_lo;

            alpha_j = abs(alpha_c - alpha_lo) < abs(alpha_q - alpha_lo) ?
                      alpha_c : alpha_c + (alpha_q - alpha_c) / 2;

            bracketed = true;

        } else if (f_i <= f_lo && df_i.dot(p) * df_lo.dot(p) < 0) {
            double z = 3 * (f_lo - f_i) / (alpha_i - alpha_lo) + df_lo.dot(p) + df_i.dot(p);
            double w = sqrt(z * z - df_lo.dot(p) * df_i.dot(p));

            if (alpha_i > alpha_lo) {
                w *= -1;
            }
            double alpha_c = (w - df_i.dot(p) + z) * (alpha_lo - alpha_i);
            alpha_c /= (df_lo.dot(p) - df_i.dot(p) + 2 * w);
            alpha_c += alpha_i;

            double alpha_s = df_i.dot(p) * (alpha_lo - alpha_i);
            alpha_s /= df_i.dot(p) - df_lo.dot(p);
            alpha_s += alpha_i;

            alpha_j = abs(alpha_c - alpha_i) > abs(alpha_s - alpha_i) ?
                      alpha_c : alpha_s;

            bracketed = true;

        } else if (f_i <= f_lo && df_i.dot(p) * df_lo.dot(p) >= 0 && abs(df_i.dot(p)) < abs(df_lo.dot(p))) {
            double z = 3 * (f_lo - f_i) / (alpha_i - alpha_lo) + df_lo.dot(p) + df_i.dot(p);
            double w = sqrt(z * z - df_lo.dot(p) * df_i.dot(p));

            if (alpha_i > alpha_lo) {
                w *= -1;
            }

            double r = (w - df_i.dot(p) + z) / (df_lo.dot(p) - df_i.dot(p) + 2 * w);
            double alpha_c;

            if (r < 0 && w != 0) {
                alpha_c = alpha_i + r * (alpha_lo - alpha_i);
            } else if (alpha_i > alpha_lo) {
                alpha_c = a_max;
            } else {
                alpha_c = a_min;
            }

            double alpha_s = df_i.dot(p) * (alpha_lo - alpha_i);
            alpha_s /= df_i.dot(p) - df_lo.dot(p);
            alpha_s += alpha_i;

            if (bracketed) {
                alpha_j = abs(alpha_c - alpha_i) < abs(alpha_s - alpha_i) ?
                          alpha_c : alpha_s;
                alpha_j = alpha_i > alpha_lo ? fmin(alpha_j, alpha_i + 0.66 * (alpha_hi - alpha_i)) :
                          fmax(alpha_j, alpha_i + 0.66 * (alpha_hi - alpha_i));
            } else {
                alpha_j = abs(alpha_c - alpha_i) > abs(alpha_s - alpha_i) ?
                          alpha_c : alpha_s;
                alpha_j = fmin(alpha_j, a_max);
                alpha_j = fmax(alpha_j, a_min);
            }

        } else {  // f_i <= f_lo && df_i * df_lo >= 0 && abs(df_i) > abs(df_lo)
            if (bracketed) {
                double z = 3 * (f_i - f_hi) / (alpha_hi - alpha_i) + df_hi.dot(p) + df_i.dot(p);
                double w = sqrt(z * z - df_hi.dot(p) * df_i.dot(p));

                if (alpha_i > alpha_hi) {
                    w *= -1;
                }

                double alpha_c = (w - df_i.dot(p) + z) * (alpha_hi - alpha_i);
                alpha_c /= (df_hi.dot(p) - df_i.dot(p) + 2 * w);
                alpha_c += alpha_i;

                alpha_j = alpha_c;
            } else if (alpha_i > alpha_lo) {
                alpha_j = a_max;
            } else {
                alpha_j = a_min;
            }
        }

        // Update the interval which contains a minimizer.
        if (f_i > f_lo) {
            alpha_hi = alpha_i;
            f_hi = f_i;
            df_hi = df_i;
        } else {
            if (df_i.dot(p) * df_lo.dot(p) < 0) {
                alpha_hi = alpha_lo;
                f_hi = f_lo;
                df_hi = df_lo;
            }
            alpha_lo = alpha_i;
            f_lo = f_i;
            df_lo = df_i;
        }

        return alpha_j;
    }

    // This subroutine finds a step
    // that satisfies a sufficient decrease condition and a curvature condition.
    // Each call of the subroutine updates an interval with endpoints alpha_lo and alpha_hi.
    double search(double& alpha_i, double f_i, VectorXd df_i,
            double& alpha_lo, double& f_lo, VectorXd& df_lo,
            double& alpha_hi, double& f_hi, VectorXd& df_hi,
            const double f_0, const VectorXd& df_0,
            size_t& stage, bool& bracketed, double& width, double& width_1,
            double& a_min, double& a_max) {
        // f(alpha_{low})      | f_lo
        // f(alpha_{high})     | f_hi
        // df(alpha_{low})     | df_lo
        // alpha_{quadratic}   | alpha_i

        // Warning tests
        if (bracketed && (alpha_i <= a_min || alpha_i >= a_max)) {
            param.result = LSResult::WARNING;
            std::cerr << "WARNING: rounding errors prevent progress" << std::endl;
        } if (bracketed && a_max - a_min <= param.xtol * a_max) {
            param.result = LSResult::WARNING;
            std::cerr << "WARNING: xtol test satisfied" << std::endl;
        } if (alpha_i == a_max
        && f_i <= f_0 + param.c_1 * alpha_i * df_0.dot(p)
        && df_i.dot(p) <= param.c_1 * df_0.dot(p)) {
            param.result = LSResult::WARNING;
            std::cerr << "WARNING: alpha = a_max" << std::endl;
        } if (alpha_i == a_min
        && (f_i > f_0 + param.c_1 * alpha_i * df_0.dot(p) || df_i.dot(p) > param.c_1 * df_0.dot(p))) {
            param.result = LSResult::WARNING;
            std::cerr << "WARNING: alpha = a_min" << std::endl;
        }

        if (f_i <= f_0 + param.c_1 * alpha_i * df_0.dot(p)
            && abs(df_i.dot(p)) <= -param.c_2 * df_0.dot(p)) {
            param.result = LSResult::CONVERGED;
        }

        switch(param.result) {
            case LSResult::CONVERGED:
                return alpha_i;
            case LSResult::WARNING:
                return alpha_i;
            default:
                break;
        }

        if (stage == 1 && f_i - f_0 - param.c_1 * alpha_i * df_0.dot(p) <= 0 && df_i.dot(p) > 0) {
            stage = 2;
            param.result = LSResult::PHI;  // stage 2
        } else if (stage == 1 && f_i < f_lo && f_i - f_0 - param.c_1 * alpha_i * df_0.dot(p) > 0) {
            param.result = LSResult::PSI;  // still stage 1
            f_i = f_i - f_0 - param.c_1 * alpha_i * df_0.dot(p);
            df_i = df_i - df_0 - param.c_1 * df_0;

            f_lo = f_lo - f_0 - param.c_1 * alpha_lo * df_0.dot(p);
            df_lo = df_lo - df_0 - param.c_1 * df_0;

            f_hi = f_hi - f_0 - param.c_1 * alpha_hi * df_0.dot(p);
            df_hi = df_hi - df_0 - param.c_1 * df_0;

            // do we really need to restore values afterall?? YES
        }

        double alpha_j = step(alpha_lo, f_lo, df_lo,
                alpha_hi, f_hi, df_hi,
                alpha_i, f_i, df_i,
                bracketed,
                a_min, a_max);

        // Reset the function and derivative values for f.
        switch(param.result) {
            case LSResult::PSI:
                f_lo = f_lo + f_0 + alpha_lo * param.c_1 * df_0.dot(p);
                df_lo = df_lo + df_0 + param.c_1 * df_0;

                f_hi = f_hi + f_0 + alpha_hi * param.c_1 * df_0.dot(p);
                df_hi = df_hi + df_0 + param.c_1 * df_0;
                break;
            default:
                break;
        }

        // CORRECTION
        // Decide if a bisection step is needed.
        if (bracketed) {
            if (abs(alpha_hi - alpha_lo) >= 0.66 * width_1) {
                alpha_j = alpha_lo + (alpha_hi - alpha_lo) / 2;
            }
            width_1 = width;
            width = abs(alpha_hi - alpha_lo);
        }

        // Set the minimum and maximum steps allowed for stp.
        if (bracketed) {
            a_min = fmin(alpha_lo, alpha_hi);
            a_max = fmax(alpha_lo, alpha_hi);
        } else {
            a_min = alpha_j + param.xtrapl * (alpha_j - alpha_lo);
            a_max = alpha_j + param.xtrapu * (alpha_j - alpha_lo);
        }

        // Force the step to be within the bounds stpmax and stpmin.
        alpha_j = fmax(alpha_j, param.alpha_min);
        alpha_j = fmin(alpha_j, param.alpha_max);

        // If further progress is not possible,
        // let stp be the best point obtained during the search.
        if (bracketed && (alpha_j <= a_min || alpha_j >= a_max)
        || bracketed && a_max - a_min <= param.xtol * a_max) {
            alpha_j = alpha_lo;
        }

        param.result = LSResult::FG;
        return alpha_j;
    }

    // That's for call from BFGS
    double get_alpha(double prev) {
        try {
            bracketing(prev);
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Error: " << ia.what() << std::endl;
        }
        return param.alpha;
    }
};

#endif //CW_LINESEARCH_H
