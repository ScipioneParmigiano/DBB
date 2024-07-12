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

#include "optioncleaner.hpp"

Eigen::Matrix<bool, Eigen::Dynamic, 1>
OptionCleaner::getFeasibleOptionFlags ( const Eigen::VectorXd& sp,
                                        const Eigen::VectorXd& bid,
                                        const Eigen::VectorXd& ask,
                                        const Eigen::VectorXd& strike,
                                        const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
                                        double spotsP, double spbid, double spask ) {
    
    M_Model M = new mosek::fusion::Model ( "FeasibleOptionFlags" );
        // M->setLogHandler ( [=] ( const std::string & msg ) {
        //     std::cout << msg << std::flush;
        // } );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } );
    
    const double SCALER ( sp.size() );

    Eigen::VectorXd lb = bid.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;
    Eigen::VectorXd ub = ask.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;

    unsigned int OPTLEN ( strike.size() );
    unsigned int LEN ( sp.size() );


    
    Eigen::VectorXd    payoffMat(OPTLEN*LEN);
    // now fill the payoff matrix
    // scale everything by mid prices so that it becomes of order 1
    for ( size_t i = 0; i < OPTLEN; ++i ) {
        for ( size_t j = 0; j < LEN; ++j ) {
            payoffMat[i * LEN + j] = otm_payoff (  sp[j], strike[i], pFlag[i] )/(0.5*(bid[i]+ask[i])*spotsP);
        }
    }
    const M_Matrix::t payoff_wrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2 ( payoffMat.data(), monty::shape ( OPTLEN, LEN ) ) ) );

    M_Variable::t q_vars = M->variable("q_vars", LEN, M_Domain::inRange(0.0, SCALER));
    M_Variable::t optVars = M->variable("optVars", OPTLEN, M_Domain::binary());
    auto lb_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( lb.data(), monty::shape ( OPTLEN) )) ;
    auto ub_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( ub.data(), monty::shape ( OPTLEN) )) ;
     
    // M_Expression::t lb_expr = M_Expr::mulElm(lb_wrap, q_vars); // std::unique_ptr<GRBVar[]> q_vars ( model.addVars ( qlo.data(), qhi.data(), NULL, NULL, NULL, LEN ) );
    // std::unique_ptr<GRBVar[]>  optVars ( model.addVars ( binlo.data(), binhi.data(), NULL, binType.data(), NULL, OPTLEN ) );

    // for ( size_t i = 0; i < OPTLEN; ++i ) {
    //     GRBLinExpr lhs = 0;
    //     for ( size_t j = 0; j < LEN; ++j ) {
    //         if ( payoffMat[i * LEN + j]>0.0 ) {
    //             lhs += payoffMat[i * LEN + j]*q_vars[j];
    //         }
    //     }
    //     model.addConstr ( lhs, GRB_GREATER_EQUAL, lb[i] * optVars[i] );
    //     model.addConstr ( lhs, GRB_LESS_EQUAL, ub[i] * optVars[i] + ( 1.0 - optVars[i] ) * SCALER * spotsP );
    // }
    M->constraint(M_Expr::sub(M_Expr::mul(payoff_wrap, q_vars), M_Expr::mulElm(lb_wrap,optVars)),M_Domain::greaterThan(0.0));
    M->constraint(M_Expr::sub(M_Expr::mul(payoff_wrap, q_vars), M_Expr::add(M_Expr::mulElm(ub_wrap,optVars),M_Expr::mul(optVars, -SCALER*spotsP))),M_Domain::lessThan( SCALER * spotsP));


    // GRBLinExpr q_lhs = 0;
    // for ( size_t j = 0; j < LEN; ++j ) {
    //     q_lhs += q_vars[j];
    // }
    // model.addConstr ( q_lhs,  GRB_EQUAL, SCALER );
    
    M->constraint(M_Expr::sum(q_vars), M_Domain::equalsTo( SCALER));
    auto sp_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1(LEN, std::function<double(ptrdiff_t)>( [&](ptrdiff_t i) { return sp(i); } ))); //

    // now the forward pricing constraints
    M->constraint(M_Expr::mul(1.0/spotsP, M_Expr::dot(sp_wrap,q_vars)),M_Domain::inRange(SCALER*spbid,SCALER*spask));


    M->objective( mosek::fusion::ObjectiveSense::Maximize, M_Expr::sum(optVars));
    M->solve();
    Eigen::Matrix<bool, Eigen::Dynamic, 1> outOpt;
    if (M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
        auto sol = optVars->level();
        const Eigen::Map<Eigen::VectorXd> solWrap(sol->raw(), OPTLEN);
        outOpt = solWrap.unaryExpr([](double arg){return arg >= 1.0 ? true : false;});
  }
  else {
        std::cout << "infeasible " <<  std::endl;
        exit(0);
    }

    return(outOpt);
}

Eigen::VectorXd OptionCleaner::getMidPriceQ ( const Eigen::VectorXd& sp,
                                        const Eigen::VectorXd& bid,
                                        const Eigen::VectorXd& ask,
                                        const Eigen::VectorXd& strike,
                                        const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
                                        double spotsP, double spbid, double spask) {
    M_Model M = new mosek::fusion::Model ( "MidPriceQ" );
        // M->setLogHandler ( [=] ( const std::string & msg ) {
        //     std::cout << msg << std::flush;
        // } );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } );
   const double SCALER ( sp.size() );

    Eigen::VectorXd lb = bid.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;
    Eigen::VectorXd ub = ask.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;

    unsigned int OPTLEN ( strike.size() );
    unsigned int LEN ( sp.size() );


    
    Eigen::VectorXd    payoffMat(OPTLEN*LEN);
    // now fill the payoff matrix
    // scale everything by mid prices so that it becomes of order 1
    for ( size_t i = 0; i < OPTLEN; ++i ) {
        for ( size_t j = 0; j < LEN; ++j ) {
            payoffMat[i * LEN + j] = otm_payoff (  sp[j], strike[i], pFlag[i] )/(0.5*(bid[i]+ask[i])*spotsP);
        }
    }
    const M_Matrix::t payoff_wrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2 ( payoffMat.data(), monty::shape ( OPTLEN, LEN ) ) ) );

    M_Variable::t q_vars = M->variable("q_vars", LEN, M_Domain::inRange(0.0, SCALER));
    auto lb_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( lb.data(), monty::shape ( OPTLEN) )) ;
    auto ub_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( ub.data(), monty::shape ( OPTLEN) )) ;
    M_Variable::t options = M->variable("optVars", OPTLEN, M_Domain::inRange(lb_wrap, ub_wrap));
    M->constraint(M_Expr::sub(M_Expr::mul(payoff_wrap, q_vars),options), M_Domain::equalsTo(0.0));
    
    M_Variable::t uu1 = M->variable(M_Domain::greaterThan(0.0));
    M_Variable::t uu2 = M->variable(M_Domain::greaterThan(0.0));
    
    M->constraint("uu1", M_Expr::vstack(0.5, uu1, M_Expr::sub(options,lb_wrap)), M_Domain::inRotatedQCone()); // quadratic cone for objective function
    M->constraint("uu2", M_Expr::vstack(0.5, uu2, M_Expr::sub(options,ub_wrap)), M_Domain::inRotatedQCone()); // quadratic cone for objective function

    
    M->constraint(M_Expr::sum(q_vars), M_Domain::equalsTo( SCALER));
    auto sp_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1(LEN, std::function<double(ptrdiff_t)>( [&](ptrdiff_t i) { return sp(i); } ))); //

    // now the forward pricing constraints
    M->constraint(M_Expr::mul(1.0/spotsP, M_Expr::dot(sp_wrap,q_vars)),M_Domain::inRange(SCALER*spbid,SCALER*spask));


    M->objective( mosek::fusion::ObjectiveSense::Minimize, M_Expr::add(uu1,uu2));
    M->solve();
    Eigen::VectorXd outOpt;
    if (M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
        auto sol = q_vars->level();
        const Eigen::Map<Eigen::VectorXd> solWrap(sol->raw(), LEN);
        outOpt = solWrap/SCALER;
  }
  else {
        std::cout << "infeasible " <<  std::endl;
        exit(0);
    }
    return(outOpt);
}


Eigen::VectorXd OptionCleaner::getMidPriceQReg ( const Eigen::VectorXd& sp,
                                        const Eigen::VectorXd& bid,
                                        const Eigen::VectorXd& ask,
                                        const Eigen::VectorXd& strike,
                                        const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
                                        double spotsP, double spbid, double spask) {
    M_Model M = new mosek::fusion::Model ( "MidPriceQReg" );
        // M->setLogHandler ( [=] ( const std::string & msg ) {
        //     std::cout << msg << std::flush;
        // } );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } );
   const double SCALER ( sp.size() );

    Eigen::VectorXd lb = bid.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;
    Eigen::VectorXd ub = ask.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;

    unsigned int OPTLEN ( strike.size() );
    unsigned int LEN ( sp.size() );


    
    Eigen::VectorXd    payoffMat(OPTLEN*LEN);
    // now fill the payoff matrix
    // scale everything by mid prices so that it becomes of order 1
    for ( size_t i = 0; i < OPTLEN; ++i ) {
        for ( size_t j = 0; j < LEN; ++j ) {
            payoffMat[i * LEN + j] = otm_payoff (  sp[j], strike[i], pFlag[i] )/(0.5*(bid[i]+ask[i])*spotsP);
        }
    }
    const M_Matrix::t payoff_wrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2 ( payoffMat.data(), monty::shape ( OPTLEN, LEN ) ) ) );

    M_Variable::t q_vars = M->variable("q_vars", LEN, M_Domain::inRange(0.0, SCALER));
    auto lb_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( lb.data(), monty::shape ( OPTLEN) )) ;
    auto ub_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( ub.data(), monty::shape ( OPTLEN) )) ;
    M_Variable::t options = M->variable("optVars", OPTLEN, M_Domain::inRange(lb_wrap, ub_wrap));
    M->constraint(M_Expr::sub(M_Expr::mul(payoff_wrap, q_vars),options), M_Domain::equalsTo(0.0));
    
    M_Variable::t uu1 = M->variable(M_Domain::greaterThan(0.0));
    M_Variable::t uu2 = M->variable(M_Domain::greaterThan(0.0));
    
    M->constraint("uu1", M_Expr::vstack(0.5, uu1, M_Expr::sub(options,lb_wrap)), M_Domain::inRotatedQCone()); // quadratic cone for objective function
    M->constraint("uu2", M_Expr::vstack(0.5, uu2, M_Expr::sub(options,ub_wrap)), M_Domain::inRotatedQCone()); // quadratic cone for objective function

    
    M->constraint(M_Expr::sum(q_vars), M_Domain::equalsTo( SCALER));
    auto sp_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1(LEN, std::function<double(ptrdiff_t)>( [&](ptrdiff_t i) { return sp(i); } ))); 
    auto sp_wrap_8 = std::shared_ptr<M_ndarray_1> (new M_ndarray_1(LEN, std::function<double(ptrdiff_t)>( [&](ptrdiff_t i) { return std::pow(sp(i)/spotsP-1.0,8); } ))); //

    // now the forward pricing constraints
    M->constraint(M_Expr::mul(1.0/spotsP, M_Expr::dot(sp_wrap,q_vars)),M_Domain::inRange(SCALER*spbid,SCALER*spask));


    M->objective( mosek::fusion::ObjectiveSense::Minimize, M_Expr::add(M_Expr::add(uu1,uu2),M_Expr::dot(q_vars,sp_wrap_8)));
    M->solve();
    Eigen::VectorXd outOpt;
    if (M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
        auto sol = q_vars->level();
        const Eigen::Map<Eigen::VectorXd> solWrap(sol->raw(), LEN);
        outOpt = solWrap/SCALER;
  }
  else {
        std::cout << "infeasible " <<  std::endl;
        exit(0);
    }
    return(outOpt);
}


Eigen::VectorXd OptionCleaner::getQReg ( const Eigen::VectorXd& sp,
                                        const Eigen::VectorXd& bid,
                                        const Eigen::VectorXd& ask,
                                        const Eigen::VectorXd& strike,
                                        const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
                                        double spotsP, double spbid, double spask) {
    M_Model M = new mosek::fusion::Model ( "QReg" );
        // M->setLogHandler ( [=] ( const std::string & msg ) {
        //     std::cout << msg << std::flush;
        // } );
        auto _M = monty::finally ( [&]() {
            M->dispose();
        } );
   const double SCALER ( sp.size() );

    Eigen::VectorXd lb = bid.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;
    Eigen::VectorXd ub = ask.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;

    unsigned int OPTLEN ( strike.size() );
    unsigned int LEN ( sp.size() );


    
    Eigen::VectorXd    payoffMat(OPTLEN*LEN);
    // now fill the payoff matrix
    // scale everything by mid prices so that it becomes of order 1
    for ( size_t i = 0; i < OPTLEN; ++i ) {
        for ( size_t j = 0; j < LEN; ++j ) {
            payoffMat[i * LEN + j] = otm_payoff (  sp[j], strike[i], pFlag[i] )/(0.5*(bid[i]+ask[i])*spotsP);
        }
    }
    const M_Matrix::t payoff_wrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2 ( payoffMat.data(), monty::shape ( OPTLEN, LEN ) ) ) );

    M_Variable::t q_vars = M->variable("q_vars", LEN, M_Domain::inRange(0.0, SCALER));
    auto lb_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( lb.data(), monty::shape ( OPTLEN) )) ;
    auto ub_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( ub.data(), monty::shape ( OPTLEN) )) ;
    M_Variable::t options = M->variable("optVars", OPTLEN, M_Domain::inRange(lb_wrap, ub_wrap));
    M->constraint(M_Expr::sub(M_Expr::mul(payoff_wrap, q_vars),options), M_Domain::equalsTo(0.0));
    
    M_Variable::t uu1 = M->variable(M_Domain::greaterThan(0.0));
    M_Variable::t uu2 = M->variable(M_Domain::greaterThan(0.0));
    
    M->constraint("uu1", M_Expr::vstack(0.5, uu1, q_vars), M_Domain::inRotatedQCone()); // quadratic cone for objective function
    // M->constraint("uu2", M_Expr::vstack(0.5, uu2, M_Expr::sub(options,ub_wrap)), M_Domain::inRotatedQCone()); // quadratic cone for objective function

    
    M->constraint(M_Expr::sum(q_vars), M_Domain::equalsTo( SCALER));
    auto sp_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1(LEN, std::function<double(ptrdiff_t)>( [&](ptrdiff_t i) { return sp(i); } ))); 
    auto sp_wrap_8 = std::shared_ptr<M_ndarray_1> (new M_ndarray_1(LEN, std::function<double(ptrdiff_t)>( [&](ptrdiff_t i) { return std::pow(sp(i)/spotsP-1.0,8); } ))); //

    // now the forward pricing constraints
    M->constraint(M_Expr::mul(1.0/spotsP, M_Expr::dot(sp_wrap,q_vars)),M_Domain::inRange(SCALER*spbid,SCALER*spask));


    M->objective( mosek::fusion::ObjectiveSense::Minimize, M_Expr::add(M_Expr::mul(0.000005,uu1),M_Expr::dot(q_vars,sp_wrap_8)));
    M->solve();
    Eigen::VectorXd outOpt;
    if (M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
        auto sol = q_vars->level();
        const Eigen::Map<Eigen::VectorXd> solWrap(sol->raw(), LEN);
        outOpt = solWrap/SCALER;
  }
  else {
        std::cout << "infeasible " <<  std::endl;
        exit(0);
    }
    return(outOpt);
}



// // // // // 
// OLD GUORBI 
// // // // // 


// Eigen::Matrix<bool, Eigen::Dynamic, 1>
// OptionCleaner::getFeasibleOptionFlags ( const Eigen::VectorXd& sp,
//                                         const Eigen::VectorXd& bid,
//                                         const Eigen::VectorXd& ask,
//                                         const Eigen::VectorXd& strike,
//                                         const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
//                                         double spotsP, double spbid, double spask ) {
//     // std::unique_ptr<GRBEnv> env =  make_unique<GRBEnv>();
//     // env->set ( GRB_IntParam_OutputFlag,0 );
//     // GRBModel model = GRBModel ( *env );
//     
//     M_Model M = new mosek::fusion::Model ( "FeasibleOptionFlags" );
// //         M->setLogHandler ( [=] ( const std::string & msg ) {
// //             std::cout << msg << std::flush;
// //         } );
//         auto _M = monty::finally ( [&]() {
//             M->dispose();
//         } );
//     
//     const double SCALER ( sp.size() );
// 
//     Eigen::VectorXd lb = bid.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;
//     Eigen::VectorXd ub = ask.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;
// 
//     unsigned int OPTLEN ( strike.size() );
//     unsigned int LEN ( sp.size() );
// 
//     // const Eigen::VectorXd binlo = Eigen::VectorXd::Constant ( OPTLEN,0.0 );
//     // const Eigen::VectorXd binhi = Eigen::VectorXd::Constant ( OPTLEN,1.0 );
//     // const Eigen::Matrix<char, Eigen::Dynamic, 1> binType = Eigen::Matrix<char, Eigen::Dynamic, 1>::Constant ( OPTLEN,GRB_BINARY );
// 
//     // const Eigen::VectorXd qlo = Eigen::VectorXd::Constant ( LEN,0.0 );
//     // const Eigen::VectorXd qhi = Eigen::VectorXd::Constant ( LEN,SCALER );
//     
//     Eigen::VectorXd    payoffMat(OPTLEN*LEN);
// //     now fill the payoff matrix
// //     scale everything by mid prices so that it becomes of order 1
//     for ( size_t i = 0; i < OPTLEN; ++i ) {
//         for ( size_t j = 0; j < LEN; ++j ) {
//             payoffMat[i * LEN + j] = otm_payoff (  sp[j], strike[i], pFlag[i] )/(0.5*(bid[i]+ask[i])*spotsP);
//         }
//     }
//     const M_Matrix::t payoff_wrap = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> ( new M_ndarray_2 ( payoffMat.data(), monty::shape ( OPTLEN, LEN ) ) ) );
// 
//     M_Variable::t q_vars = M->variable("q_vars", LEN, M_Domain::inRange(0.0, SCALER));
//     M_Variable::t optVars = M->variable("optVars", OPTLEN, M_Domain::binary());
//     auto lb_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( lb.data(), monty::shape ( OPTLEN) )) ;
//     auto ub_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( ub.data(), monty::shape ( OPTLEN) )) ;
//     // M_Expression::t lb_expr = M_Expr::mulElm(lb_wrap, q_vars);
//     // std::unique_ptr<GRBVar[]> q_vars ( model.addVars ( qlo.data(), qhi.data(), NULL, NULL, NULL, LEN ) );
//     // std::unique_ptr<GRBVar[]>  optVars ( model.addVars ( binlo.data(), binhi.data(), NULL, binType.data(), NULL, OPTLEN ) );
// 
//     // for ( size_t i = 0; i < OPTLEN; ++i ) {
//     //     GRBLinExpr lhs = 0;
//     //     for ( size_t j = 0; j < LEN; ++j ) {
//     //         if ( payoffMat[i * LEN + j]>0.0 ) {
//     //             lhs += payoffMat[i * LEN + j]*q_vars[j];
//     //         }
//     //     }
//     //     model.addConstr ( lhs, GRB_GREATER_EQUAL, lb[i] * optVars[i] );
//     //     model.addConstr ( lhs, GRB_LESS_EQUAL, ub[i] * optVars[i] + ( 1.0 - optVars[i] ) * SCALER * spotsP );
//     // }
//     M->constraint(M_Expr::sub(M_Expr::mul(payoff_wrap, q_vars), M_Expr::mulElm(lb_wrap,optVars)),M_Domain::greaterThan(0.0));
//     M->constraint(M_Expr::sub(M_Expr::mul(payoff_wrap, q_vars), M_Expr::add(M_Expr::mulElm(ub_wrap,optVars),M_Expr::mul(optVars, -SCALER*spotsP))),M_Domain::lessThan( SCALER * spotsP));
// 
// 
//     // GRBLinExpr q_lhs = 0;
//     // for ( size_t j = 0; j < LEN; ++j ) {
//     //     q_lhs += q_vars[j];
//     // }
//     // model.addConstr ( q_lhs,  GRB_EQUAL, SCALER );
//     
//     M->constraint(M_Expr::sum(q_vars), M_Domain::equalsTo( SCALER));
//     
//     auto sp_wrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( sp.data(), monty::shape ( LEN) )) ;
// 
// //     now the forward pricing constraints
//     GRBLinExpr sp_forw = 0;
//     for ( size_t j = 0; j < LEN; ++j ) {
//         sp_forw += sp[j]/spotsP * q_vars[j];
//     }
//     model.addConstr ( sp_forw,  GRB_LESS_EQUAL, SCALER*spask );
//     model.addConstr ( sp_forw,  GRB_GREATER_EQUAL, SCALER*spbid );
//     M->constraint(M_Expr::sum(q_vars), M_Domain::equalsTo( SCALER));
// 
// 
//     GRBLinExpr obj = 0;
//     for ( size_t j = 0; j < OPTLEN; ++j ) {
//         obj += optVars[j];
//     }
//     model.setObjective ( obj,GRB_MAXIMIZE );
//     model.optimize();
// 
//     Eigen::Matrix<bool, Eigen::Dynamic, 1> outOpt(OPTLEN);
// //     std::ofstream feasible("feasible.txt",std::fstream::in | std::fstream::out | std::fstream::app);
//     if ( model.get ( GRB_IntAttr_Status ) == GRB_OPTIMAL || model.get ( GRB_IntAttr_Status ) == GRB_SUBOPTIMAL ) {
//         for ( size_t i = 0; i < OPTLEN; i++ ) {
//             if ( std::abs ( optVars[i].get ( GRB_DoubleAttr_X ) ) >=1.0 ) { // keep the option
//                 outOpt[i] = true;
//             } else {
//                 outOpt[i] = false;
//             }
// //             std::cout << optVars[i].get(GRB_DoubleAttr_X) << std::endl;
//         }
// 
//     } else {
//         std::cout << "infeasible " <<  std::endl;
//         exit(0);
//     }
// 
//     return(outOpt);
// }
// 
// Eigen::VectorXd OptionCleaner::getMidPriceQ ( const Eigen::VectorXd& sp,
//                                         const Eigen::VectorXd& bid,
//                                         const Eigen::VectorXd& ask,
//                                         const Eigen::VectorXd& strike,
//                                         const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
//                                         double spotsP, double spbid, double spask) {
//         std::unique_ptr<GRBEnv> env =  make_unique<GRBEnv>();
//     env->set ( GRB_IntParam_OutputFlag,0 );
//     GRBModel model = GRBModel ( *env );
//     const double SCALER ( sp.size() );
// 
//     const Eigen::VectorXd lb = bid.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;
//     const Eigen::VectorXd ub = ask.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;
// 
//     unsigned int OPTLEN ( strike.size() );
//     unsigned int LEN ( sp.size() );
// 
//     const Eigen::VectorXd qlo = Eigen::VectorXd::Constant ( LEN,0.0 );
//     const Eigen::VectorXd qhi = Eigen::VectorXd::Constant ( LEN,SCALER );
//     
//     Eigen::VectorXd    payoffMat(OPTLEN*LEN);
// //     now fill the payoff matrix
// //     scale everything by mid prices so that it becomes of order 1
//     for ( size_t i = 0; i < OPTLEN; ++i ) {
//         for ( size_t j = 0; j < LEN; ++j ) {
//             payoffMat[i * LEN + j] = otm_payoff (  sp[j], strike[i], pFlag[i] )/(0.5*(bid[i]+ask[i])*spotsP);
//         }
//     }
// 
// 
//     std::unique_ptr<GRBVar[]> q_vars ( model.addVars ( qlo.data(), qhi.data(), NULL, NULL, NULL, LEN ) );
//     std::unique_ptr<GRBVar[]> options ( model.addVars ( lb.data(), ub.data(), NULL, NULL, NULL, OPTLEN ) );
//     
//     GRBQuadExpr obj = 0;
// 
//     for ( size_t i = 0; i < OPTLEN; ++i ) {
//         GRBLinExpr lhs = 0;
//         for ( size_t j = 0; j < LEN; ++j ) {
//             if ( payoffMat[i * LEN + j]>0.0 ) {
//                 lhs += payoffMat[i * LEN + j]*q_vars[j];
//             }
//         }
//         model.addConstr ( lhs, GRB_EQUAL, options[i]);
//         obj += (options[i]-lb[i])*(options[i]-lb[i])+(options[i]-ub[i])*(options[i]-ub[i]);
//     }
// 
// 
//     GRBLinExpr q_lhs = 0;
//     for ( size_t j = 0; j < LEN; ++j ) {
//         q_lhs += q_vars[j];
//     }
//     model.addConstr ( q_lhs,  GRB_EQUAL, SCALER );
// 
// //     now the forward pricing constraints
//     GRBLinExpr sp_forw = 0;
//     for ( size_t j = 0; j < LEN; ++j ) {
//         sp_forw += sp[j]/spotsP * q_vars[j];
//     }
//     model.addConstr ( sp_forw,  GRB_LESS_EQUAL, SCALER*spask );
//     model.addConstr ( sp_forw,  GRB_GREATER_EQUAL, SCALER*spbid );
// 
// 
// 
//     model.setObjective ( obj,GRB_MINIMIZE );
//     model.optimize();
// 
//     Eigen::VectorXd outOpt(LEN);
// //     std::ofstream feasible("feasible.txt",std::fstream::in | std::fstream::out | std::fstream::app);
//     if ( model.get ( GRB_IntAttr_Status ) == GRB_OPTIMAL  ) {
//         for ( size_t i = 0; i < LEN; i++ ) {
//             outOpt(i) = q_vars[i].get ( GRB_DoubleAttr_X )/SCALER;
//         }
// 
//     } else {
//         std::cout << "infeasible " <<  std::endl;
//         exit(0);
//     }
// 
//     return(outOpt);
// }
// 
// 
// Eigen::VectorXd OptionCleaner::getMidPriceQReg ( const Eigen::VectorXd& sp,
//                                         const Eigen::VectorXd& bid,
//                                         const Eigen::VectorXd& ask,
//                                         const Eigen::VectorXd& strike,
//                                         const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
//                                         double spotsP, double spbid, double spask) {
//         std::unique_ptr<GRBEnv> env =  make_unique<GRBEnv>();
//     env->set ( GRB_IntParam_OutputFlag,0 );
//     GRBModel model = GRBModel ( *env );
//     const double SCALER ( sp.size() );
// 
//     const Eigen::VectorXd lb = bid.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;
//     const Eigen::VectorXd ub = ask.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;
// 
//     unsigned int OPTLEN ( strike.size() );
//     unsigned int LEN ( sp.size() );
// 
//     const Eigen::VectorXd qlo = Eigen::VectorXd::Constant ( LEN,0.0 );
//     const Eigen::VectorXd qhi = Eigen::VectorXd::Constant ( LEN,SCALER );
//     
//     Eigen::VectorXd    payoffMat(OPTLEN*LEN);
// //     now fill the payoff matrix
// //     scale everything by mid prices so that it becomes of order 1
//     for ( size_t i = 0; i < OPTLEN; ++i ) {
//         for ( size_t j = 0; j < LEN; ++j ) {
//             payoffMat[i * LEN + j] = otm_payoff (  sp[j], strike[i], pFlag[i] )/(0.5*(bid[i]+ask[i])*spotsP);
//         }
//     }
// 
// 
//     std::unique_ptr<GRBVar[]> q_vars ( model.addVars ( qlo.data(), qhi.data(), NULL, NULL, NULL, LEN ) );
//     std::unique_ptr<GRBVar[]> options ( model.addVars ( lb.data(), ub.data(), NULL, NULL, NULL, OPTLEN ) );
//     
//     GRBQuadExpr obj = 0;
// 
//     for ( size_t i = 0; i < OPTLEN; ++i ) {
//         GRBLinExpr lhs = 0;
//         for ( size_t j = 0; j < LEN; ++j ) {
//             if ( payoffMat[i * LEN + j]>0.0 ) {
//                 lhs += payoffMat[i * LEN + j]*q_vars[j];
//             }
//         }
//         model.addConstr ( lhs, GRB_EQUAL, options[i]);
//         obj += (options[i]-lb[i])*(options[i]-lb[i])+(options[i]-ub[i])*(options[i]-ub[i]);
//     }
// 
// 
//     GRBLinExpr q_lhs = 0;
//     for ( size_t j = 0; j < LEN; ++j ) {
//         q_lhs += q_vars[j];
//     }
//     model.addConstr ( q_lhs,  GRB_EQUAL, SCALER );
// 
// //     now the forward pricing constraints
//     GRBLinExpr sp_forw = 0;
//     GRBLinExpr sp_fourthmom = 0;
//     for ( size_t j = 0; j < LEN; ++j ) {
//         sp_forw += sp[j]/spotsP * q_vars[j];
//         sp_fourthmom += pow(sp[j]/spotsP-1.0,8) * q_vars[j];
//     }
//     model.addConstr ( sp_forw,  GRB_LESS_EQUAL, SCALER*spask );
//     model.addConstr ( sp_forw,  GRB_GREATER_EQUAL, SCALER*spbid );
// 
// 
// 
//     model.setObjective ( obj+sp_fourthmom,GRB_MINIMIZE );
//     model.optimize();
// 
//     Eigen::VectorXd outOpt(LEN);
// //     std::ofstream feasible("feasible.txt",std::fstream::in | std::fstream::out | std::fstream::app);
//     if ( model.get ( GRB_IntAttr_Status ) == GRB_OPTIMAL  ) {
//         for ( size_t i = 0; i < LEN; i++ ) {
//             outOpt(i) = q_vars[i].get ( GRB_DoubleAttr_X )/SCALER;
//         }
// 
//     } else {
//         std::cout << "infeasible " <<  std::endl;
//         exit(0);
//     }
// 
//     return(outOpt);
// }
// 
// 
// Eigen::VectorXd OptionCleaner::getQReg ( const Eigen::VectorXd& sp,
//                                         const Eigen::VectorXd& bid,
//                                         const Eigen::VectorXd& ask,
//                                         const Eigen::VectorXd& strike,
//                                         const Eigen::Matrix<bool, Eigen::Dynamic, 1>& pFlag,
//                                         double spotsP, double spbid, double spask) {
//         std::unique_ptr<GRBEnv> env =  make_unique<GRBEnv>();
//     env->set ( GRB_IntParam_OutputFlag,0 );
//     GRBModel model = GRBModel ( *env );
//     const double SCALER ( sp.size() );
// 
//     const Eigen::VectorXd lb = bid.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;
//     const Eigen::VectorXd ub = ask.cwiseQuotient ( 0.5 * ( bid+ask ) *spotsP ) * SCALER;
// 
//     unsigned int OPTLEN ( strike.size() );
//     unsigned int LEN ( sp.size() );
// 
//     const Eigen::VectorXd qlo = Eigen::VectorXd::Constant ( LEN,0.0 );
//     const Eigen::VectorXd qhi = Eigen::VectorXd::Constant ( LEN,SCALER );
//     
//     Eigen::VectorXd    payoffMat(OPTLEN*LEN);
// //     now fill the payoff matrix
// //     scale everything by mid prices so that it becomes of order 1
//     for ( size_t i = 0; i < OPTLEN; ++i ) {
//         for ( size_t j = 0; j < LEN; ++j ) {
//             payoffMat[i * LEN + j] = otm_payoff (  sp[j], strike[i], pFlag[i] )/(0.5*(bid[i]+ask[i])*spotsP);
//         }
//     }
// 
// 
//     std::unique_ptr<GRBVar[]> q_vars ( model.addVars ( qlo.data(), qhi.data(), NULL, NULL, NULL, LEN ) );
//     std::unique_ptr<GRBVar[]> options ( model.addVars ( lb.data(), ub.data(), NULL, NULL, NULL, OPTLEN ) );
//     
//     GRBQuadExpr obj = 0;
// 
//     for ( size_t i = 0; i < OPTLEN; ++i ) {
//         GRBLinExpr lhs = 0;
//         for ( size_t j = 0; j < LEN; ++j ) {
//             if ( payoffMat[i * LEN + j]>0.0 ) {
//                 lhs += payoffMat[i * LEN + j]*q_vars[j];
//             }
//         }
//         model.addConstr ( lhs, GRB_EQUAL, options[i]);
// //         obj += (options[i]-lb[i])*(options[i]-lb[i])+(options[i]-ub[i])*(options[i]-ub[i]);
//     }
// 
// 
//     GRBLinExpr q_lhs = 0;
//     for ( size_t j = 0; j < LEN; ++j ) {
//         q_lhs += q_vars[j];
//     }
//     model.addConstr ( q_lhs,  GRB_EQUAL, SCALER );
// 
// //     now the forward pricing constraints
//     GRBLinExpr sp_forw = 0;
//     GRBLinExpr sp_fourthmom = 0;
//     for ( size_t j = 0; j < LEN; ++j ) {
//         sp_forw += sp[j]/spotsP * q_vars[j];
//         sp_fourthmom += pow(sp[j]/spotsP-1.0,8) * q_vars[j];
//         obj +=  0.000005*q_vars[j]*q_vars[j];
//     }
//     model.addConstr ( sp_forw,  GRB_LESS_EQUAL, SCALER*spask );
//     model.addConstr ( sp_forw,  GRB_GREATER_EQUAL, SCALER*spbid );
// 
// 
// 
//     model.setObjective ( obj+sp_fourthmom,GRB_MINIMIZE );
//     model.optimize();
// 
//     Eigen::VectorXd outOpt(LEN);
// //     std::ofstream feasible("feasible.txt",std::fstream::in | std::fstream::out | std::fstream::app);
//     if ( model.get ( GRB_IntAttr_Status ) == GRB_OPTIMAL  ) {
//         for ( size_t i = 0; i < LEN; i++ ) {
//             outOpt(i) = q_vars[i].get ( GRB_DoubleAttr_X )/SCALER;
//         }
// 
//     } else {
//         std::cout << "infeasible " <<  std::endl;
//         exit(0);
//     }
// 
//     return(outOpt);
// }
// 




