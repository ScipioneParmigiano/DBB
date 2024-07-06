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
    double alpha = 1e4;  // Example value of alpha
    std::vector<int> omega_l = {0, 1, 2};  // Example mask omega_l: select the states of interest

    // example payoffs, TODO: compute payoff from data
    std::vector<std::vector<double>> payoff_matrix = {
        {11.0, 8.0, 9.0},
        {20.0, 18.0, 29.0}
    };
    std::vector<double> ask = {10.0 ,20.0};
    std::vector<double> bid = {9.0, 19.0};

    // Initialization
    Model::t M = new Model("main");
    auto _M = finally([&]() { M->dispose(); });

    // Define variables P and Q
    Variable::t p = M->variable("P", n, Domain::greaterThan(0.0));  // Variable P
    Variable::t q = M->variable("Q", n, Domain::greaterThan(0.0));  // Variable Q

    // Add constraints (make the P and Q congruent distributions)
    M->constraint(Expr::sum(p), Domain::equalsTo(1.0));  // Sum of p elements equals 1
    M->constraint(Expr::sum(q), Domain::equalsTo(1.0));  // Sum of q elements equals 1

    // Add additional constraints involving payoff_matrix and q
    for (int i = 0; i < m; ++i) {
        auto payoff_row = new_array_ptr<double>(payoff_matrix[i]);
        Expression::t dot_product = Expr::dot(payoff_row, q); // Dot product of payoff row and q
        
        M->constraint("bid_" + std::to_string(i), dot_product, Domain::greaterThan(bid[i])); // Adding bid constraint
        M->constraint("ask_" + std::to_string(i), dot_product, Domain::lessThan(ask[i])); // Adding ask constraint
    }

    // Create ancillary variables for the second moment of the pricing kernel constraint
    Variable::t q_square = M->variable(n, Domain::greaterThan(0.0));
    Variable::t p_square = M->variable(n, Domain::greaterThan(0.0));    
    Variable::t one = M->variable(1, Domain::equalsTo(1.0));
    
    for(int i=0; i<n; ++i){
        M->constraint("q_square" + std::to_string(i), Expr::hstack(Expr::mul(q_square->index(i), 0.5), one, q->index(i)), Domain::inRotatedQCone());
        M->constraint("p_square" + std::to_string(i), Expr::hstack(Expr::mul(p_square->index(i), 0.5), one, p->index(i)), Domain::inRotatedQCone());
    }

    // minimize q_square so that q_square = q*q (same for p)
    Expression::t obj_q_square = Expr::add(Expr::sum(q_square), Expr::sum(p_square));
    M->objective("obj_q_square", ObjectiveSense::Minimize, obj_q_square);

    // Solve to get the equality
    M->solve();

    // Second moment of the pricing kernel constraint
    for(int i=0; i<n; ++i){
        M->constraint("second moment of the pricing kernel" + std::to_string(i), Expr::sub(Expr::mul(1/alpha, q_square->index(i)), p->index(i)), Domain::lessThan(0.0));
    }

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
