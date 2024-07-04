#include <iostream>
#include <vector>
#include "fusion.h"
#include <random>
#include <cmath>
#include "/home/pietro/Desktop/USI/RRCA/RRCA/src/util/Macros.h"

// using namespace RRCA;
using namespace mosek::fusion;
using namespace monty;

int main() {
    int n = 3;  // Example dimension
    int m = 2;   // Number of assets
    double alpha = 1;  // Example value of alpha
    std::vector<int> omega_l = {0,1, 2};  // Example mask omega_l: select the states of interest

    // std::vector<std::vector<double>> payoff_matrix(m, std::vector<double>(n, 1.0)); // Example matrix, initialized to ones
    // std::vector<std::vector<double>> payoff_matrix(m, std::vector<double>(n));

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // double a = 0.1;
    // std::uniform_real_distribution<double> dis(a, a);

    //  for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         payoff_matrix[i][j] = dis(gen);
    //     }
    // }

    // example payoffs, TODO: compute payoff from data
    std::vector<std::vector<double>> payoff_matrix = {
        {11.0, 8.0, 9.0},
        {20.0, 18.0, 29.0}
    };
    std::vector<double> ask = {10.0 ,20.0};
    std::vector<double> bid = {9.0, 19.0};

    Model::t M = new Model("example");
    auto _M = finally([&]() { M->dispose(); }); // Automatically dispose the model at the end

    // Define variables
    Variable::t p = M->variable("P", n, Domain::greaterThan(0.0));  // Variable P
    Variable::t q = M->variable("Q", n, Domain::greaterThan(0.0));  // Variable Q

    // Add constraints (make the P and Q congruent distributions)
    M->constraint(Expr::sum(p), Domain::equalsTo(1.0));  // Sum of p elements equals 1
    M->constraint(Expr::sum(q), Domain::equalsTo(1.0));  // Sum of q elements equals 1

    // Add additional constraints involving payoff_matrix and q
    for (int i = 0; i < m; ++i) {
        auto payoff_row = new_array_ptr<double>(payoff_matrix[i]);
        Expression::t dot_product = Expr::dot(payoff_row, q); // Dot product of payoff row and q

        std::cout << i << std::endl;

        // std::cout << "Dot product of payoff_matrix row " << i << " and q: " << dot_product->level() << std::endl;

        
        M->constraint("bid_" + std::to_string(i), dot_product, Domain::greaterThan(bid[i])); // Adding bid constraint
        M->constraint("ask_" + std::to_string(i), dot_product, Domain::lessThan(ask[i])); // Adding ask constraint
    }


    typedef mosek::fusion::Variable M_Variable;
    typedef mosek::fusion::Domain M_Domain;
    typedef mosek::fusion::Expression M_Expression; 
    typedef mosek::fusion::Expr M_Expr; 

    M_Variable::t uu = M->variable(n, M_Domain::greaterThan(0.0)); // the moment matrix
    M_Variable::t pp = M->variable(n, M_Domain::greaterThan(0.0)); // the diagonal
    M_Variable::t qq = M->variable(n, M_Domain::greaterThan(0.0)); // the diagonal
    M->constraint(M_Expr::hstack(M_Expr::mul(uu,0.5), qq, pp), M_Domain::inRotatedQCone());
    M->constraint(M_Expr::sum(uu), M_Domain::lessThan(alpha)); 

    // // Second moment constrain: introduce an auxiliary variable
    // auto t = M->variable("t", Domain::greaterThan(0.0));

    // // Second-order cone constraint: \sum p_i^2 <= alpha * \sum q_i
    std::vector<Variable::t> p_square;
    auto p_square_i = M->variable("P_SQ_" /*+ std::to_string(i)*/, n, Domain::greaterThan(0.0)); // Auxiliary variable for p[i]^2
    for (int i = 0; i < n; ++i) {
        M->constraint("cone_" + std::to_string(i), Expr::sub(p_square_i->index(i), Expr::eval(p->index(i)), Domain::equalsTo(0.0)); // p_square_i = p[i]
        p_square.push_back(p_square_i);
    }

    //     std::vector<Variable::t> p_square;
    // auto p_square_i = M->variable("P_SQ_" /*+ std::to_string(i)*/, n, Domain::greaterThan(0.0)); // Auxiliary variable for p[i]^2
    // for (int i = 0; i < n; ++i) {
    //     M->constraint("cone_" + std::to_string(i), Expr::sub(p_square_i->index(i), p->index(i)), Domain::equalsTo(0.0)); // p_square_i = p[i]
    //     p_square.push_back(p_square_i);
    // }

    // // Construct sum expressions for p^2 and alpha * q
    // Expression::t sum_p_square = p_square[0];
    // for (int i = 1; i < n; ++i) {
    //     sum_p_square = Expr::add(sum_p_square, p_square[i]);
    // }p_square

    // Expression::t var_p_expr = Expr::constTerm(0.0);
    // Expression::t var_q_expr = Expr::constTerm(0.0);
    // for (int i = 0; i < n; ++i) {
    //     var_p_expr = Expr::add(var_p_expr, Expr::mul(p_square[i], 1.0 / n)); // Var(p) = sum(p_i^2) / n
    //     var_q_expr = Expr::add(var_q_expr, Expr::mul(q_square[i], 1.0 / n)); // Var(q) = sum(q_i^2) / n
    // }
    // M->constraint("variance_constraint", Expr::sub(var_p_expr, var_q_expr), Domain::greaterThan(0.0));

    // Define objective function using mask omega_l
    Expression::t obj_expr = Expr::constTerm(0.0);
    for (size_t i = 0; i < omega_l.size(); ++i) {
        int idx = omega_l[i];
        obj_expr = Expr::add(obj_expr, Expr::add(p->index(idx), q->index(idx)));
    }

    M->objective("obj", ObjectiveSense::Maximize, obj_expr);

    // Solve the problem
    M->solve();
    


    // Print solution
    auto sol_p = p->level();
    auto sol_q = q->level();
    std::cout << "Solution:" << std::endl;
    std::cout << "P = " << *sol_p << std::endl;
    std::cout << "Q = " << *sol_q << std::endl;

    auto sol_ps = qq->level();
    std::cout << "P^2 = " << *sol_ps << std::endl;

    // Print dot products after optimization
    std::cout << "Dot products after optimization:" << std::endl;
    for (int i = 0; i < m; ++i) {
        auto payoff_row = payoff_matrix[i];
        double dot_product_value = 0.0;
        for (int j = 0; j < n; ++j) {
            dot_product_value += payoff_row[j] * (*sol_q)[j];
        }
        std::cout << "Dot product of payoff_matrix row " << i << " and q: " << dot_product_value << std::endl;
    }


    return 0;
}
