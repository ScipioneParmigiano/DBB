/*
 * Copyright 2021 <copyright holder> <email>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OPTIONCLEANER_H
#define OPTIONCLEANER_H

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <eigen3/Eigen/Dense>
#include "fusion.h"

namespace OptionCleaner {
    typedef mosek::fusion::Matrix M_Matrix; 

typedef mosek::fusion::Variable M_Variable; 

typedef mosek::fusion::Var M_Var; 

typedef mosek::fusion::Expression M_Expression; 

typedef mosek::fusion::Domain M_Domain;

typedef monty::ndarray<double, 1> M_ndarray_1;

typedef monty::ndarray<double, 2> M_ndarray_2;

typedef mosek::fusion::Expr M_Expr; 

typedef mosek::fusion::Model::t M_Model; 
    
    template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

  inline double  otm_payoff(double state, double strike, bool pFlag)  {
    if(pFlag){
        return(std::max(strike - state,0.0));
    }
    return(std::max(state - strike,0.0));
    }
    
    
    // returns vector of boolean flags which options to keep in order for NA given the state space
    Eigen::Matrix<bool,Eigen::Dynamic,1> getFeasibleOptionFlags(const Eigen::VectorXd& sp_, 
           const Eigen::VectorXd& bid_, 
           const Eigen::VectorXd& ask_, 
           const Eigen::VectorXd& strike_, 
           const Eigen::Matrix<bool,Eigen::Dynamic,1>& pFlag_, 
           double spotsP_,double spbid, double spask);
    
    // returns vector of martingale probabilities that generate options as close as possible to midprices
    Eigen::VectorXd getMidPriceQ(const Eigen::VectorXd& sp_, 
           const Eigen::VectorXd& bid_, 
           const Eigen::VectorXd& ask_, 
           const Eigen::VectorXd& strike_, 
           const Eigen::Matrix<bool,Eigen::Dynamic,1>& pFlag_, 
           double spotsP_,double spbid, double spask);
    
    // returns vector of martingale probabilities that generate options as close as possible to midprices
    Eigen::VectorXd getMidPriceQReg(const Eigen::VectorXd& sp_, 
           const Eigen::VectorXd& bid_, 
           const Eigen::VectorXd& ask_, 
           const Eigen::VectorXd& strike_, 
           const Eigen::Matrix<bool,Eigen::Dynamic,1>& pFlag_, 
           double spotsP_,double spbid, double spask);
    
    //  returns vector of martingale probabilities that generate options as close as possible to midprices
    Eigen::VectorXd getQReg(const Eigen::VectorXd& sp_, 
           const Eigen::VectorXd& bid_, 
           const Eigen::VectorXd& ask_, 
           const Eigen::VectorXd& strike_, 
           const Eigen::Matrix<bool,Eigen::Dynamic,1>& pFlag_, 
           double spotsP_,double spbid, double spask);


}

#endif // OPTIONCLEANER_H
