#ifndef CW_PARAM_H
#define CW_PARAM_H

#include <Eigen/Core>

// Line Search Result
// – a flag that describes the process
enum class LSResult {
    CONVERGED = 1,
    WARNING = 2,
    PSI = 3,
    PHI = 4,
    FG = 5
};

using Eigen::VectorXd;

/// Set of parameters to control BFGS algorithm
class BFGSParam {
public:
    /// Maximum iterations of BFGS
    size_t max_iter;

    /// Maximum number of line search iterations
    size_t max_line_search;

    /// Convergence tolerance
    double epsilon;

    /// Initial step. 1 is the recommended initial value
    double alpha;

    /// Step value bounds. Values are taken from Python scalar_search_wolfe_1
    double alpha_max;
    double alpha_min;

    /// Wolfe parameter c_1: 0 < c_1 < c_2 < 1
    double c_1;

    /// Wolfe parameter c_2: 0 < c_1 < c_2 < 1
    double c_2;

    /// Relative tolerance for an acceptable step.
    double xtol;

    /// WHAT
    double xtrapl;
    double xtrapu;

    /// Flag
    LSResult result;

    BFGSParam() {
        max_iter = 100;
        max_line_search = 20;
        epsilon = 1e-6;
        alpha = 1;
        alpha_max = 1e100;
        alpha_min = 1e-100;
        c_1 = 1e-4;
        c_2 = 0.9;
        xtol = 1e-14;
        xtrapl = 1.1;
        xtrapu = 4.0;
        result = LSResult::FG;
    }

    inline void check_param() const
    {
        if(max_iter <= 0)
            throw std::invalid_argument("'max_iter' must be positive");
        if(max_line_search <= 0)
            throw std::invalid_argument("'max_line_search' must be positive");
        if(epsilon < 0)
            throw std::invalid_argument("'epsilon' must be non-negative");
        if(alpha < 0)
            throw std::invalid_argument("'alpha' must be positive");
        if(alpha_min < 0 || alpha_min >= alpha_max)
            throw std::invalid_argument("'alpha_min' must satisfy 0 < alpha_min < alpha_max");
        if(alpha_max <= alpha_min)
            throw std::invalid_argument("'alpha_max' must be greater than alpha_min");
        if (c_1 <= 0 || c_1 >= c_2)
            throw std::invalid_argument("'c_1' must satisfy 0 < c_1 < c_2");
        if (c_2 <= c_1 || c_2 >= 1)
            throw std::invalid_argument("'c_2' must satisfy c_1 < c_2 < 1");
        if(xtol < 0)
            throw std::invalid_argument("'xtol' must be non-negative");
    }
};

#endif //CW_PARAM_H
