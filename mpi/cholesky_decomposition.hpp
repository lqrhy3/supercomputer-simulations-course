#ifndef CHOLESKY_DECOMPOSITION_HPP
#define CHOLESKY_DECOMPOSITION_HPP

#include "array_operations.hpp"

Matrix<double> decompose_cholesky(Matrix<double>& original, MPI_Comm comm, int comm_size, int root_rank) {

    Matrix<double> triangular(original.nrows(), original.ncols());
    for (ptrdiff_t i = 0; i < triangular.nrows(); i++) {
        for (ptrdiff_t j = i + 1; j < triangular.ncols(); j++) {
            triangular(i, j) = 0;
        }
    }

    for (ptrdiff_t j = 0; j < triangular.ncols(); j++) {
        double sum = 0;
        for (ptrdiff_t t = 0; t < j; t++) {
            sum += pow(triangular(j, t), 2);
        }

        triangular(j, j) = sqrt(original(j, j) - sum);
        if (j == 0) {
            for (ptrdiff_t i = j + 1; i < triangular.nrows(); i++) {
                triangular(i, j) = original(i, j) / triangular(j, j);
            }
        } else {
            int nrows = original.nrows() - (j + 1), ncols = j;
            Matrix<double> matrix = extract_submatrix(triangular, j);
            Vec<double> vec(ncols), product(nrows);

            for (ptrdiff_t k = 0; k < ncols; k++) {
                vec(k) = triangular(j, k);
            }

            matvec_mpi(matrix.raw_ptr(), vec.raw_ptr(), product.raw_ptr(), nrows, ncols, comm, comm_size, root_rank);

            for (ptrdiff_t i = j + 1; i < triangular.nrows(); i++) {
                triangular(i, j) = (original(i, j) - product(i - j - 1)) / triangular(j, j);
            }
        }
    }

    return triangular;
}

Matrix<double> decompose_cholesky_block(Matrix<double> base, MPI_Comm comm, int comm_size, int root_rank, int size = 100) {
    Matrix<double> result(base.nrows(), base.ncols());
    int iterations = base.nrows() / size;
    bool fully_block = (base.nrows() % size) == 0;

    for (ptrdiff_t i = 0; i < result.nrows(); i++) {
        for (ptrdiff_t j = i + 1; j < result.ncols(); j++) {
            result(i, j) = 0;
        }
    }

    for (int k = 0; k < iterations; k++) {
        Matrix<double> square(size, size), band(base.nrows() - size, size), residual(base.nrows() - size, base.nrows() - size);
        tie(square, band, residual) = split_base_block(base, size);

        Matrix<double> triangular = decompose_cholesky(square, comm, comm_size, root_rank);

        int shift = size * k;
        Matrix<double> inverted = invert_triangular(band, triangular);
        insert_result_block(result, triangular, inverted, shift);

        if  ((k == iterations - 1) and fully_block)
            break;


        Matrix<double> product(inverted.nrows(), inverted.nrows());

        MPI_Barrier(comm);
        matmul_transposed_mpi(inverted.raw_ptr(), inverted.raw_ptr(), product.raw_ptr(),
                              inverted.nrows(), inverted.ncols(), inverted.nrows(), comm, comm_size, root_rank);

        MPI_Barrier(comm);
        matsubtract_inplace_mpi(residual.raw_ptr(), product.raw_ptr(),
                                residual.nrows(), residual.ncols(), comm, comm_size, root_rank);

        MPI_Barrier(comm);
        base = residual;

        if (k == iterations - 1) {
            Matrix<double> triangular = decompose_cholesky(base, comm, comm_size, root_rank);
            for (ptrdiff_t i = 0; i < triangular.nrows(); i++) {
                for (ptrdiff_t j = 0; j < triangular.ncols(); j++) {
                    result(i + shift + size, j + shift + size) = triangular(i, j);
                }
            }
        }


    }

    return result;
}

#endif
