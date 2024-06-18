/**
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef GLMNET_EIGEN_WRAPPER_H
#define GLMNET_EIGEN_WRAPPER_H

#include "tasks/cox_regression/coxnet.h"
#include <vector>

namespace Eigen {
    typedef Matrix<int, Eigen::Dynamic, 1> VectorXuint8;
}

template <typename T, typename S, typename U>
int fit_coxnet(
        Eigen::Map<T>& x_map,
        Eigen::Map<S>& time_map,
        Eigen::Map<U>& event_map,
        Eigen::Map<S>& pen_map,
        Eigen::Map<S>& alphas_map,
        Eigen::Map<T>& coef_path_map,
        Eigen::Map<S>& final_alphas_map,
        Eigen::Map<S>& final_dev_ratio_map,
        bool create_path,
        typename T::Scalar alpha_min_ratio,
        typename T::Scalar l1_ratio,
        std::size_t max_iter,
        double eps,
        bool verbose)
{
    typedef Eigen::Map<T> MatrixType;
    typedef Eigen::Map<S> VectorType;
    typedef Eigen::Map<U> IntVectorType;
    typedef coxnet::Coxnet<MatrixType, VectorType, IntVectorType> CoxnetType;
    typedef typename CoxnetType::DataType DataType;
    typedef coxnet::FitResult<MatrixType, VectorType> ResultType;

    const DataType _data(x_map, time_map, event_map, pen_map);
    ResultType result(coef_path_map, final_alphas_map, final_dev_ratio_map);

    const coxnet::Parameters _params(alpha_min_ratio, l1_ratio, max_iter, eps, verbose);
    CoxnetType object(_data, _params);
    object.fit(alphas_map, create_path, result);

    return result.getNumberOfAlphas();
}

#endif //GLMNET_EIGEN_WRAPPER_H