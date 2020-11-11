#include <Eigen/Core>
#include <cmath>
#include <thread>
#include "BFGS.h"
#include "Approximation.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Array;

class LogLikelihood
{
private:
    // m rows and n columns
    // matrix of covariates x is (m x n) and target y is (m x 1)
    // k (alpha_0 ... alpha_k)
    int m, n, k;
    MatrixXd x;
    VectorXd y;
    // l2 regularization
    double a = 1.0;
    CDF& cdf;

public:
    LogLikelihood(int m_, int n_, int k_, MatrixXd x_, VectorXd  y_, CDF& cdf_)
            : m(m_), n(n_), k(k_), x(std::move(x_)), y(std::move(y_)), cdf(cdf_) {}

    // calculate LogLikelihood at theta and update gradient
    double operator() (const VectorXd& theta, VectorXd& grad) {
        // I assume that the first k elements of theta vector are alpha params
        cdf.reset_param(theta.head(k));
        // LogLikelihood
        double fx = 0.0;
        // every row
        for (int i = 0; i < m; i++) {
            std::pair <double, VectorXd> cdf_xt = cdf(x.block<1, 3>(i, 0), theta.tail(n));
            fx += -y[i] * log(cdf_xt.first) - (1 - y[i]) * log(1 - cdf_xt.first);

            // update gradient
            grad = grad + (cdf_xt.first - y[i]) / (cdf_xt.first * (1 - cdf_xt.first)) * cdf_xt.second;
        }
        std::chrono::milliseconds timespan(50);
        std::this_thread::sleep_for(timespan);
        return fx + a * theta.squaredNorm() / 2;
    }
};

class Logit
{
public:
    Logit(int m_, int n_, MatrixXd x_, VectorXd  y_)
            : m(m_), n(n_), x(std::move(x_)), y(std::move(y_)) {}

    VectorXd fit() {
        // Set up parameters
        BFGSParam param;

        VectorXd alpha = VectorXd::Ones(3);
        CDF dist(alpha);

        // Create solver and function object
        BFGS solver(param);
        LogLikelihood f(m, n, 3, x, y, dist);

        // Initial guess
        VectorXd theta(n + 3);
        theta << 5, 1, 3, 1, 1, 0.06;
        // theta will be overwritten to be the best point found
        solver.minimize(f, theta);

        std::cout << solver.k << " iterations" << std::endl;
        std::cout << "theta = " << theta.transpose() << std::endl;
        Result r = solver.get_result();
        switch(r) {
            case Result::CONVERGED:
                std::cout << "Converged" << std::endl;
                break;
            case Result::NOT_CONVERGED:
                std::cout << "Has not converged" << std::endl;
        }

        return theta;
    }

private:
    int m, n;
    MatrixXd x;
    VectorXd y;
};

int main() {
    int m = 30, n = 3;
    MatrixXd x(m, n);
    VectorXd y(m);
    x << 580.0, 2.3, 2.0,
            650.0, 2.3, 1.0,
            580.0, 3.3, 1.0,
            690.0, 3.3, 3.0,
            690.0, 3.7, 5.0,
            640.0, 3.0, 1.0,
            680.0, 3.3, 5.0,
            580.0, 2.7, 4.0,
            670.0, 2.7, 2.0,
            740.0, 3.3, 5.0,
            680.0, 3.3, 4.0,
            730.0, 3.7, 6.0,
            650.0, 3.7, 6.0,
            770.0, 3.3, 3.0,
            660.0, 3.3, 6.0,
            720.0, 3.3, 4.0,
            660.0, 4.0, 4.0,
            750.0, 3.9, 4.0,
            660.0, 3.7, 4.0,
            710.0, 3.7, 6.0,
            620.0, 2.7, 2.0,
            570.0, 3.0, 2.0,
            690.0, 2.3, 1.0,
            550.0, 2.7, 1.0,
            600.0, 2.0, 1.0,
            590.0, 2.3, 3.0,
            690.0, 1.7, 1.0,
            590.0, 1.7, 4.0,
            710.0, 3.7, 5.0,
            780.0, 4.0, 3.0;
    y << 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1;

    Logit logit(m, n, x, y);
    VectorXd theta = logit.fit();

    MatrixXd x_(10, 3);
    x_ << 550.0, 2.3, 4.0,
            620.0, 3.3, 2.0,
            670.0, 3.3, 6.0,
            680.0, 3.9, 4.0,
            610.0, 2.7, 3.0,
            610.0, 3.0, 1.0,
            650.0, 3.7, 6.0,
            690.0, 3.7, 5.0,
            540.0, 2.7, 2.0,
            660.0, 3.3, 5.0;

    VectorXd y_(10);
    for (int i = 0; i < 10; i++) {
        y_[i] = exp(x_.block<1, 3>(i, 0).dot(theta));
    }
    VectorXd ans(10);
    ans << 0, 1, 1, 0, 0, 0, 1, 1, 0, 1;
    double acc = 0.0;
    for (int i = 0; i < 10; i++) {
        int a = y_[i] / (1 + y_[i]) >= 0.5 ? 1 : 0;
        std::cout << a << ' ';
        acc += ans[i] == a ? 1 : 0;
    }
    std::cout << std::endl << "Accuracy: " << acc / 10;
    return 0;
}
