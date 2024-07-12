#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "fusion.h"

namespace py = pybind11;

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> numpy_mat_to_eigen(py::array_t<T, py::array::c_style | py::array::forcecast> arr);

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> numpy_vec_to_eigen(py::array_t<T, py::array::c_style | py::array::forcecast> arr);

double otm_payoff(double spot, double strike, bool isPut);

std::shared_ptr<monty::ndarray<double, 1>> omegaLMask(const Eigen::VectorXi& positions, int n);

template<typename T>
std::shared_ptr<monty::ndarray<T, 1>> eigenToStdVector(const Eigen::VectorXd& eigenVec);

template<typename T>
std::shared_ptr<monty::ndarray<T, 2>> eigenToStdMatrix(const Eigen::MatrixXd& eigenMat);

std::vector<double> computeGrossReturns(const Eigen::MatrixXd& payoff_matrix);

py::tuple performOptimization(int n, double alpha, double lambda,
                              py::array_t<int> omega_l_py,
                              py::array_t<double> sp_py,
                              py::array_t<double> strike_py,
                              py::array_t<double> bid_py,
                              py::array_t<double> ask_py,
                              py::array_t<int> pFlag_py);

#endif
