#ifndef CW_BFGS_H
#define CW_BFGS_H

#include <iostream>
#include "LineSearch.h"

using Eigen::VectorXd, Eigen::MatrixXd;

// x vector is given by the link at the function call and
// is changed during the algorithm work

enum class Result {
    CONVERGED,
    NOT_CONVERGED
};

// First goes the name I give then the object
class BFGS {
private:
    const BFGSParam& param;
    Result result;

public:
    size_t k = 0;

    explicit BFGS(const BFGSParam& param_) : param(param_)
    {
        param.check_param();
    }

    template <typename F, template<class> class LineSearch = LineSearch>
    void minimize(F& f, VectorXd &x) {
        size_t n = x.size();

        // Notation
        /// current Hessian approximation
        MatrixXd H = MatrixXd::Identity(n, n);

        /// gradient value {i} and {i + 1}
        VectorXd df = VectorXd::Zero(n);

        /// x_{i + 1} - x_{i},
        /// gradient_{i + 1} - gradient_{i} and
        /// descent direction
        VectorXd s(n);
        VectorXd y(n);
        VectorXd p(n);

        double f_c;
        double f_p;

        // Calculate the value of the objective function and update the gradient
        // we must not care about what we have in gradient vector now
        f_c = f(x, df);

        /// Sets the initial step guess to dx ~ 1???
        f_p = f_c + df.norm() / 2;

        for (;;) {
            // Convergence test
            // If the number of iterations reaches the limit,
            // algorithms terminates
            if (df.norm() <= param.epsilon) {
                result = Result::CONVERGED;
                return;
            } else if (k >= param.max_iter) {
                result = Result::NOT_CONVERGED;
                return;
            }

            // Calculate descent direction
            p = -H * df;

            // Calculate step alpha
            // First I create a class
            // make it calculate step size
            // and return it
            LineSearch line_search(f, x, p, param);
            double alpha = line_search.get_alpha(f_p);
            alpha = fmin(alpha, 1e-2);

            // Define vector s_{i}, update x
            x += alpha * p;
            s = alpha * p;

            // Define y_{i}, update gradient
            y = -df;
            df.setZero();
            f_p = f_c;
            f_c = f(x, df);
            y += df;

            // If no update has been done
            // We change the current matrix first
//            if (k == 0)
//            {
//                H = y.dot(s) / (y.squaredNorm()) * MatrixXd::Identity(x.size(), x.size());
//            }

            // Update the matrix
            double rho = 1 / (y.dot(s));
            H = (MatrixXd::Identity(x.size(), x.size()) - rho * s * y.transpose()) *
                H * (MatrixXd::Identity(x.size(), x.size())
                     - rho * y * s.transpose()) + rho * s * s.transpose();
            k++;
        }
    }

    Result get_result() {
        return result;
    }
};

#endif //CW_BFGS_H
