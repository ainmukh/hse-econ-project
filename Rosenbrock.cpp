//#include <Eigen/Core>
//#include <iostream>
//#include "BFGS.h"
//
//using Eigen::VectorXd;
//
//class Rosenbrock
//{
//private:
//    int n;
//public:
//    Rosenbrock(int n_) : n(n_) {}
//    double operator()(const VectorXd& x, VectorXd& grad)
//    {
//        double fx = 0.0;
//        for(int i = 0; i < n; i += 2)
//        {
//            double t1 = 1.0 - x[i];
//            double t2 = 10 * (x[i + 1] - x[i] * x[i]);
//            grad[i + 1] = 20 * t2;
//            grad[i]     = -2.0 * (x[i] * grad[i + 1] + t1);
//            fx += t1 * t1 + t2 * t2;
//        }
//        return fx;
//    }
//};
//
//int main()
//{
//
//    const int n = 10;
//    // Set up parameters
//    BFGSParam param;
//
//    // Create solver and function object
//    BFGS solver(param);
//    Rosenbrock fun(n);
//
//    // Initial guess
//    VectorXd x = VectorXd::Zero(n);
//    VectorXd xa = VectorXd::Zero(n);
//    // x will be overwritten to be the best point found
//    double fx;
//    solver.minimize(fun, x);
//
//    std::cout << solver.k << " iterations" << std::endl;
//    std::cout << "theta = " << x.transpose() << std::endl;
//    Result r = solver.get_result();
//    switch(r) {
//        case Result::CONVERGED:
//            std::cout << "Converged" << std::endl;
//            break;
//        case Result::NOT_CONVERGED:
//            std::cout << "Has not converged" << std::endl;
//    }
//    double a = fun(x, xa);
//    std::cout << "fx = " << a << std::endl;
//    return 0;
//}

//int main() {
//    double li = gsl_sf_fermi_dirac_int(5, -4);  // -Li_{5 + 1}(-exp(-4))
//    double zeta = gsl_sf_zeta_int(4);  // n != 1
//    double gamma = gsl_sf_gamma(6);
//    double tan = tanh(0.567);
//    std::cout << "Polylog -Li_{5 + 1}(-exp(-4)) = " << li << std::endl;
//    std::cout << "Riemann Zeta function zeta(4) = " << zeta << std::endl;
//    std::cout << "Gamma(6) = " << gamma << std::endl;
//    std::cout << "tanh(.567) = " << tan <<std::endl;
//    std::cout << "ln{2} = " << M_LN2 << std::endl;
//
//    VectorXd A(3);
//    A << 1, 2, 3;
//    MatrixXd B(2, 3);
//    VectorXd C(3);
//    C << 4, 5, 6;
//    std::cout << "Indexing C[1] = " << C[1] << ", indexing C(1) = " << C(1) << std::endl;
//    return 0;
//}
