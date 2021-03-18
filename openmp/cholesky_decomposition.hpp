#ifndef CHOLESKY_DECOMPOSITION_HPP
#define CHOLESKY_DECOMPOSITION_HPP

#include "array_types.hpp"
#include "array_operations.hpp"

template <class T>
tuple<Matrix<T>, Matrix<T>, Matrix<T>> splitBaseBlock(Matrix<T> &A, int n) {
    Matrix<T> A_11(n, n),
            A_21(A.nrows() - n, n),
            A_22(A.nrows() - n, A.nrows() - n);

    for (ptrdiff_t i = 0; i < A.nrows(); i++) {
        for (ptrdiff_t j = 0; j < A.ncols(); j++) {
            if (i < n) {
                if (j < n) A_11(i, j) = A(i, j);
            } else  {
                if (j < n) A_21(i - n, j) = A(i, j);
                else A_22(i - n, j - n) = A(i, j);
            }
        }
    }

    return make_tuple(A_11, A_21, A_22);
}


template <class T>
Matrix<T> getL_21(Matrix<T>& A_11, Matrix<T>& L_11) {
    Matrix<T> L_21(A_11.nrows(), L_11.nrows());

    for (ptrdiff_t j = 0; j < L_21.ncols(); j++) {
        #pragma omp parallel for
        for (ptrdiff_t i = 0; i < L_21.nrows(); i++) {
            T sum = 0;
            for (ptrdiff_t t = 0; t < j; t++) {
                sum += L_21(i, t) * L_11(j, t);
            }
            L_21(i, j) = (A_11(i, j) - sum) / L_11(j, j);
        }
    }


    return L_21;
}

template <class T>
void insertLBlocks(Matrix<T>& L, Matrix<T>& L_11, Matrix<T>& L_21, int shift) {
    int n = L_11.nrows();

    for (ptrdiff_t i = 0; i < L.nrows() - shift; i++) {
#pragma omp parallel for
        for (ptrdiff_t j = 0; j < n; j++) {
            if (i < n) L(i + shift, j + shift) = L_11(i, j);
            else L(i + shift, j + shift) = L_21(i - n, j);
        }
    }
}


template <class T>
Matrix<T> decompose_cholesky(Matrix<T>& A) {
    Matrix<double> L(A.nrows(), A.ncols());
    for (ptrdiff_t i = 0; i < L.nrows(); i++) {
        for (ptrdiff_t j = i + 1; j < L.ncols(); j++) {
            L(i, j) = 0;
        }
    }

    for (ptrdiff_t j = 0; j < L.ncols(); j++) {
        T sum = 0;
        for (ptrdiff_t t = 0; t < j; t++) {
            sum += pow(L(j, t), 2);
        }

        L(j, j) = sqrt(A(j, j) - sum);

#pragma omp parallel for
        for (ptrdiff_t i = j + 1; i < L.nrows(); i++) {
            T sum = 0;
            for (ptrdiff_t t = 0; t < j; t++) {
                sum += L(j, t) * L(i, t);
            }

            L(i, j) = (A(j, i) - sum) / L(j, j);
        }
    }

    return L;
}


template <class T>
Matrix<T> decompose_cholesky_block(Matrix<T> A, int r) {
    Matrix<T> L(A.nrows(), A.ncols());
    int nSteps = A.nrows() / r;

    for (ptrdiff_t i = 0; i < L.nrows(); i++) {
        for (ptrdiff_t j = i + 1; j < L.ncols(); j++) {
            L(i, j) = 0;
        }
    }

    for (int k = 0; k < nSteps; k++) {
        Matrix<T> A_11(r, r), A_21(A.nrows() - r, r), A_22(A.nrows() - r, A.nrows() - r);
        tie(A_11, A_21, A_22) = splitBaseBlock(A, r);

        Matrix<T> L_11 = decompose_cholesky(A_11);
        Matrix<T> L_21 = getL_21(A_21, L_11);

        int shift = r * k;
        insertLBlocks(L, L_11, L_21, shift);
        Matrix<T> prod = matmul_transposed(L_21, L_21);
        A = matsubtract_inplace(A_22, prod);

        if (k == nSteps - 1) {
            Matrix<T> L_22 = decompose_cholesky(A);
            for (ptrdiff_t i = 0; i < L_22.nrows(); i++) {
                for (ptrdiff_t j = 0; j < L_22.ncols(); j++) {
                    L(i + shift + r, j + shift + r) = L_22(i, j);
                }
            }
        }
    }

    return L;
}

#endif
