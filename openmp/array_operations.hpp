#ifndef ARRAY_OPERATIONS_H
#define ARRAY_OPERATIONS_H

#include "array_types.hpp"


template <class T>
Matrix<T> matmul_transposed(Matrix<T>& mat1, Matrix<T>& mat2) {
    Matrix<T> product(mat1.nrows(), mat2.nrows());

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < mat1.nrows(); i++) {

        for (ptrdiff_t j = 0; j < mat2.nrows(); j++) {
            product(i, j) = 0;

            for (ptrdiff_t k = 0; k < mat1.ncols(); k++) {
                product(i, j) += mat1(i, k) * mat2(j, k);
            }
        }
    }

    return product;
}

template <class T>
Matrix<T>& matsubtract_inplace(Matrix<T>& mat1, Matrix<T>& mat2) {

    // #pragma omp parallel for
    for (ptrdiff_t i = 0; i < mat1.nrows(); i++) {
#pragma omp parallel for
        for (ptrdiff_t j = 0; j < mat2.ncols(); j++) {
            mat1(i, j) -= mat2(i, j);
        }
    }

    return mat1;
}


template <class T>
bool check_decomposition(Matrix<T>& mat1, Matrix<T>& mat2, double eps = 1e-5) {
    for (ptrdiff_t i = 0; i < mat1.nrows(); i++) {
        for (ptrdiff_t j = 0; j < mat1.ncols(); j++) {
            if (abs(mat1(i, j) - mat2(i, j)) > eps) {
                return false;
            }
        }
    }
    return true;
}

#endif
