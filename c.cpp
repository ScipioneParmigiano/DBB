#include <Eigen/Dense> 
#include "fusion.h"    

// Function to compute out-of-the-money (OTM) payoff
double otm_payoff(double spot, double strike, bool isPut) {
    if (isPut) {
        return std::max(strike - spot, 0.0);
    } else {
        return std::max(spot - strike, 0.0);
    }
}

std::shared_ptr<monty::ndarray<double, 1>> omegaLMask(const Eigen::VectorXi& positions, int n) {
    // Initialize an n-long array with all elements set to 0.0
    std::vector<double> x(n);
    auto result = monty::new_array_ptr<double>(x);    
    std::fill(result->begin(), result->end(), 0.0);

    // Set specified positions to 1.0
    for (int i = 0; i < positions.size(); ++i) {
        int pos = positions(i);
        if (pos < n && pos >= 0) {  // Ensure the position is within bounds
            (*result)(pos) = 1.0;
        }
    }

    return result;
}

template<typename T>
// Create a shared_ptr for vectors of the appropriate type
std::shared_ptr<monty::ndarray<T, 1>> eigenToStdVector(const Eigen::VectorXd& eigenVec) {
    auto stdVec = std::make_shared<monty::ndarray<T, 1>>(eigenVec.data(), eigenVec.size());
    
    return stdVec;
}

template<typename T>
// Create a shared_ptr for matrices of the appropriate type
std::shared_ptr<monty::ndarray<T, 2>> eigenToStdMatrix(const Eigen::MatrixXd& eigenMat) {
    auto stdMat = std::make_shared<monty::ndarray<T, 2>>(eigenMat.data(), monty::shape(eigenMat.cols(), eigenMat.rows()));
    return stdMat;
}

// Function to compute gross returns from a payoff matrix
std::vector<double> computeGrossReturns(const Eigen::MatrixXd& payoff_matrix) {
    // Compute gross returns using Eigen's vectorized operations
    Eigen::VectorXd gross_returns = payoff_matrix.rowwise().sum();

    // Convert Eigen::VectorXd to std::vector<double>
    return std::vector<double>(gross_returns.data(), gross_returns.data() + gross_returns.size());
}

int main() {
    // Example dimensions and parameters
    int n = 4;               // Number of options
    double alpha = 1.3;      // Example value of alpha
    double lambda = 1.1;     // Example value of lambda (regularization)
    Eigen::VectorXi omega_l(2);
    omega_l << 0, 3;   // Example mask omega_l: select the states of interest

    // Example input vectors
    Eigen::VectorXd sp_eigen(n);
    sp_eigen << 1200, 1250, 1300, 1350;

    Eigen::VectorXd strike_eigen(4);
    strike_eigen << 1290, 1295, 1295, 1300;

    Eigen::VectorXd bid_eigen(4);
    bid_eigen << 27.7, 27.4, 29.4, 25.0;

    Eigen::VectorXd ask_eigen(4);
    ask_eigen << 29.3, 29.7, 31.4, 26.9;

    Eigen::VectorXi pFlag_eigen(4);
    pFlag_eigen << true, false, true, false;

    // Initialize payoff matrix and compute payoffs
    size_t spLen = sp_eigen.size();
    size_t optLen = bid_eigen.size();
    Eigen::MatrixXd payoff_matrix(optLen, spLen);


    // Fill the payoff matrix
    for (size_t i = 0; i < optLen; ++i) {
        for (size_t j = 0; j < spLen; ++j) {

            // Compute OTM payoff based on spot, strike, and option type
            payoff_matrix(i, j) = otm_payoff(sp_eigen(j), strike_eigen(i), pFlag_eigen(i)) / (0.5 * (bid_eigen[i] + ask_eigen[i]));
        }
    }

    std::cout << "Payoff Matrix:" << std::endl;
    std::cout << payoff_matrix << std::endl;



    std::vector<double> gross_returns = computeGrossReturns(payoff_matrix);
   
    // std::cout << "Computed Gross Returns:" << std::endl;
    // for (double gr : gross_returns) {
    //     std::cout << gr << " ";
    // }
    // std::cout << std::endl;

    // Initialization of the model
    mosek::fusion::Model::t M = new mosek::fusion::Model("main");
    auto _M = monty::finally([&]() { M->dispose(); });

    // Define variables P and Q
    mosek::fusion::Variable::t p = M->variable("P", n, mosek::fusion::Domain::greaterThan(0.0));  // mosek::fusion::Variable P
    mosek::fusion::Variable::t q = M->variable("Q", n, mosek::fusion::Domain::greaterThan(0.0));  // mosek::fusion::Variable Q

    // Add constraints (make the P and Q congruent distributions)
    M->constraint(mosek::fusion::Expr::sum(p), mosek::fusion::Domain::equalsTo(1.0));  // Sum of p elements equals 1
    M->constraint(mosek::fusion::Expr::sum(q), mosek::fusion::Domain::equalsTo(1.0));  // Sum of q elements equals 1

    // Add constraints involving payoff_matrix and q
    Eigen::VectorXd result_bid = bid_eigen.array() / (0.5 * (bid_eigen.array() + ask_eigen.array()));
    Eigen::VectorXd result_ask = ask_eigen.array() / (0.5 * (bid_eigen.array() + ask_eigen.array()));
    mosek::fusion::Matrix::t payoff_monty_matr = mosek::fusion::Matrix::dense(eigenToStdMatrix<double>(payoff_matrix));
    
    mosek::fusion::Expression::t product = mosek::fusion::Expr::mul(q, payoff_monty_matr);
    M->constraint("bid", product, mosek::fusion::Domain::greaterThan(eigenToStdVector<double>(result_bid)));
    M->constraint("ask", product, mosek::fusion::Domain::lessThan(eigenToStdVector<double>(result_ask)));

    // Create ancillary variables for the second moment of the pricing kernel constraint
    mosek::fusion::Variable::t q_square = M->variable(n, mosek::fusion::Domain::greaterThan(0.0));
    mosek::fusion::Variable::t p_square = M->variable(n, mosek::fusion::Domain::greaterThan(0.0));
    mosek::fusion::Variable::t one = M->variable(1, mosek::fusion::Domain::equalsTo(1.0));

    // // Constraints for second moment of pricing kernel
    // for (int i = 0; i < n; ++i) {
    //     M->constraint("q_square" + std::to_string(i), mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(q_square->index(i), 0.5), one, q->index(i)), mosek::fusion::Domain::inRotatedQCone());
    //     M->constraint("p_square" + std::to_string(i), mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(p_square->index(i), 0.5), one, p->index(i)), mosek::fusion::Domain::inRotatedQCone());
    // }

    // Constraints used for the regularization
    Eigen::VectorXd ones_vector = Eigen::VectorXd::Ones(n);
    auto ones_ptr = std::make_shared<monty::ndarray<double, 1>>(ones_vector.data(), ones_vector.size()); 
    mosek::fusion::Variable::t ones = M->variable(n, mosek::fusion::Domain::equalsTo(ones_ptr));
    M->constraint("q_square", mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(q_square, 0.5), ones, q), mosek::fusion::Domain::inRotatedQCone());
    M->constraint("p_square", mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(p_square, 0.5), ones, p), mosek::fusion::Domain::inRotatedQCone());
   

    // Second moment of the pricing kernel constraint
    mosek::fusion::Variable::t u = M->variable(n);
    M->constraint(mosek::fusion::Expr::hstack(mosek::fusion::Expr::mul(u, 0.5), p, q), mosek::fusion::Domain::inRotatedQCone());
    M->constraint(mosek::fusion::Expr::sum(u), mosek::fusion::Domain::lessThan(alpha));

    // Variance constraint using dot product
    std::shared_ptr<monty::ndarray<double, 1>> payoff (new monty::ndarray<double, 1>(n));
    for (int i = 0; i < n; ++i) {
        (*payoff)[i] = gross_returns[i] - log(gross_returns[i]) - 1;
    }

    mosek::fusion::Expression::t p_var = mosek::fusion::Expr::dot(payoff, p);
    mosek::fusion::Expression::t q_var = mosek::fusion::Expr::dot(payoff, q);

    M->constraint(mosek::fusion::Expr::sub(p_var, q_var), mosek::fusion::Domain::lessThan(0.0));

    // Useless variables and constraints, used only in this version just to asser it the variance constraint is satisfacted 
    mosek::fusion::Variable::t p_vari = M->variable(1, mosek::fusion::Domain::greaterThan(0.0));
    mosek::fusion::Variable::t q_vari = M->variable(1, mosek::fusion::Domain::greaterThan(0.0));
    M->constraint(mosek::fusion::Expr::sub(p_var, p_vari), mosek::fusion::Domain::equalsTo(0.0));
    M->constraint(mosek::fusion::Expr::sub(q_var, q_vari), mosek::fusion::Domain::equalsTo(0.0));

    // Define objective function using mask omega_l
    std::shared_ptr<monty::ndarray<double, 1>> mask = omegaLMask(omega_l, n);
    
    mosek::fusion::Expression::t obj_expr = mosek::fusion::Expr::add(mosek::fusion::Expr::dot(mask, p), mosek::fusion::Expr::dot(mask, q));
    mosek::fusion::Expression::t regularization = mosek::fusion::Expr::add(mosek::fusion::Expr::sum(p_square), mosek::fusion::Expr::sum(q_square));
    mosek::fusion::Expression::t obj_expr_reg = mosek::fusion::Expr::sub(obj_expr, mosek::fusion::Expr::mul(lambda, regularization));

    M->objective("obj", mosek::fusion::ObjectiveSense::Maximize, obj_expr_reg);

    // Solve the problem
    M->solve();

    // Print solution
    auto sol_p = p->level();
    auto sol_q = q->level();
    std::cout << "Solution:" << std::endl;
    std::cout << "P = " << *sol_p << std::endl;
    std::cout << "Q = " << *sol_q << std::endl;

    return 0;
}
