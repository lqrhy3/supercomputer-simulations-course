#ifndef ARRAY_OPERATIONS_H
#define ARRAY_OPERATIONS_H

#include "array_types.hpp"

bool check_decomposition(Matrix<double>& left_matrix, Matrix<double>& right_matrix, double eps = 1e-5) {
    for (ptrdiff_t i = 0; i < left_matrix.nrows(); i++) {
        for (ptrdiff_t j = 0; j < left_matrix.ncols(); j++) {
            double difference = abs(left_matrix(i, j) - right_matrix(i, j));
            if ((difference > eps) or (isnan(difference))) {
                 cout << "Difference " << difference << " occured at " << i << ", " << j << '\n';
                return false;
            }
        }
    }
    return true;
}

void matmul_transposed_serial(double* submatrix, double* right_matrix, double* subproduct, 
                              int nrows_each, int ntouch, int ncols) {
    
    for (ptrdiff_t i = 0; i < nrows_each; i++) {
        for (ptrdiff_t j = 0; j < ncols; j++) {
            subproduct[i * ncols + j] = 0;
            for (ptrdiff_t k = 0; k < ntouch; k++) {
                subproduct[i * ncols + j] += submatrix[i * ntouch + k] * right_matrix[j * ntouch + k];
            }
        }
    }
}

void matsubtract_inplace_serial(double* left_submatrix, double* right_submatrix, 
                                int nrows_each, int ncols) {

    for (ptrdiff_t i = 0; i < nrows_each; i++) {
        for (ptrdiff_t j = 0; j < ncols; j++) {
            left_submatrix[i * ncols + j] -= right_submatrix[i * ncols + j];
        }
    }
}

void matsubtract_inplace_mpi(double* left_matrix, double* right_matrix, int nrows, int ncols,
                             MPI_Comm comm, int comm_size, int root_rank) {

    int nrows_base = nrows / comm_size, nrows_res = nrows % comm_size;
    int displaces[comm_size], counts[comm_size];
    displaces[0] = 0;

    for (int rank = 0; rank < comm_size - 1; rank++) {
        counts[rank] = nrows_base * ncols;
        if (rank < nrows_res) {
            counts[rank] += ncols;
        }

        displaces[rank + 1] = displaces[rank] + counts[rank];
    }
    counts[comm_size - 1] = nrows_base * ncols;

    int myrank = 0;
    MPI_Comm_rank(comm, &myrank);

    double* left_submatrix = new double[counts[myrank]];
    double* right_submatrix = new double[counts[myrank]];

    MPI_Scatterv(left_matrix, counts, displaces, MPI_DOUBLE, left_submatrix, counts[myrank], MPI_DOUBLE, root_rank, comm);
    MPI_Scatterv(right_matrix, counts, displaces, MPI_DOUBLE, right_submatrix, counts[myrank], MPI_DOUBLE, root_rank, comm);

    matsubtract_inplace_serial(left_submatrix, right_submatrix, (int) (counts[myrank] / ncols), ncols);
    MPI_Gatherv(left_submatrix, counts[myrank], MPI_DOUBLE, left_matrix, counts, displaces, MPI_DOUBLE, root_rank, comm);

    delete [] left_submatrix;
    delete [] right_submatrix;
}

Matrix<double> extract_submatrix(Matrix<double> original, int index) {
    int nrows = original.nrows() - (index + 1), ncols = index;
    Matrix<double> submatrix(nrows, ncols);

    for (ptrdiff_t i = 0; i < submatrix.nrows(); i++) {
        for (ptrdiff_t j = 0; j < submatrix.ncols(); j++) {
            submatrix(i, j) = original(i + index + 1, j);
        }
    }

    return submatrix;
}

void matvec_serial(double* matrix, double* vec, double* product, int nrows, int ncols, 
                MPI_Comm comm, int comm_size, int root_rank) {
    
    for (ptrdiff_t i = 0; i < nrows; i++) {
        product[i] = 0;
        for (ptrdiff_t j = 0; j < ncols; j++) {
            product[i] += matrix[i * ncols + j] * vec[j];
        }
    }
}

void matvec_mpi(double* matrix, double* vec, double* product, int nrows, int ncols,
                MPI_Comm comm, int comm_size, int root_rank) {

    int nrows_base = nrows / comm_size, nrows_res = nrows % comm_size;
    int send_displaces[comm_size], send_counts[comm_size];
    send_displaces[0] = 0;

    int recv_displaces[comm_size], recv_counts[comm_size];
    recv_displaces[0] = 0;

    for (int rank = 0; rank < comm_size - 1; rank++) {
        send_counts[rank] = nrows_base * ncols;
        recv_counts[rank] = nrows_base;

        if (rank < nrows_res) {
            send_counts[rank] += ncols;
            recv_counts[rank] += 1;
        }

        send_displaces[rank + 1] = send_displaces[rank] + send_counts[rank];
        recv_displaces[rank + 1] = recv_displaces[rank] + recv_counts[rank];
    }
    send_counts[comm_size - 1] = nrows_base * ncols;
    recv_counts[comm_size - 1] = nrows_base;

    int myrank = 0;
    MPI_Comm_rank(comm, &myrank);

    double* submatrix = new double[send_counts[myrank]];
    double* subproduct = new double[recv_counts[myrank]];

    MPI_Bcast(vec, ncols, MPI_DOUBLE, root_rank, comm);
    MPI_Scatterv(matrix, send_counts, send_displaces, MPI_DOUBLE, submatrix,
                 send_counts[myrank], MPI_DOUBLE, root_rank, comm);

    matvec_serial(submatrix, vec, subproduct, recv_counts[myrank], ncols, comm, comm_size, root_rank);
    MPI_Gatherv(subproduct, recv_counts[myrank], MPI_DOUBLE, product, recv_counts, recv_displaces, MPI_DOUBLE, root_rank, comm);

    delete [] submatrix;
    delete [] subproduct;
}

void matmul_transposed_mpi(double* left_matrix, double* right_matrix, double* product,
                           int nrows, int ntouch, int ncols, MPI_Comm comm, int comm_size, int root_rank) {

    int nrows_base = nrows / comm_size, nrows_res = nrows % comm_size;
    int send_displaces[comm_size], send_counts[comm_size];
    send_displaces[0] = 0;

    int recv_displaces[comm_size], recv_counts[comm_size];
    recv_displaces[0] = 0;

    for (int rank = 0; rank < comm_size - 1; rank++) {
        send_counts[rank] = nrows_base * ntouch;
        recv_counts[rank] = nrows_base * ncols;

        if (rank < nrows_res) {
            send_counts[rank] += ntouch;
            recv_counts[rank] += ncols;
        }

        send_displaces[rank + 1] = send_displaces[rank] + send_counts[rank];
        recv_displaces[rank + 1] = recv_displaces[rank] + recv_counts[rank];
    }
    send_counts[comm_size - 1] = nrows_base * ntouch;
    recv_counts[comm_size - 1] = nrows_base * ncols;

    int myrank = 0;
    MPI_Comm_rank(comm, &myrank);

    double* submatrix = new double[send_counts[myrank]];
    double* subproduct = new double[recv_counts[myrank]];

    MPI_Bcast(right_matrix, ntouch * ncols, MPI_DOUBLE, root_rank, comm);
    MPI_Scatterv(left_matrix, send_counts, send_displaces, MPI_DOUBLE, submatrix,
                 send_counts[myrank], MPI_DOUBLE, root_rank, comm);

    matmul_transposed_serial(submatrix, right_matrix, subproduct,
            (int) (send_counts[myrank] / ntouch), ntouch, ncols);
    MPI_Gatherv(subproduct, recv_counts[myrank], MPI_DOUBLE, product, recv_counts, recv_displaces, MPI_DOUBLE, root_rank, comm);
    delete [] submatrix;
    delete [] subproduct;
}

Matrix<double> invert_triangular(Matrix<double>& base, Matrix<double>& triangular) {
    Matrix<double> result(base.nrows(), triangular.nrows());

    for (ptrdiff_t j = 0; j < result.ncols(); j++) {
        for (ptrdiff_t i = 0; i < result.nrows(); i++) {
            double sum = 0;
            for (ptrdiff_t t = 0; t < j; t++) {
                sum += result(i, t) * triangular(j, t);
            }
            result(i, j) = (base(i, j) - sum) / triangular(j, j);
        }
    }

    return result;
}

tuple<Matrix<double>, Matrix<double>, Matrix<double>> split_base_block(Matrix<double>& base, int size) {
    Matrix<double> square(size, size), band(base.nrows() - size, size), residual(base.nrows() - size, base.nrows() - size);
    
    for (ptrdiff_t i = 0; i < base.nrows(); i++) {
        for (ptrdiff_t j = 0; j < base.ncols(); j++) {
            if (i < size) {
                if (j < size) square(i, j) = base(i, j);
            } else  {
                if (j < size) band(i - size, j) = base(i, j);
                else residual(i - size, j - size) = base(i, j);
            }
        }
    }

    return make_tuple(square, band, residual);
}

void insert_result_block(Matrix<double>& result, Matrix<double>& triangular, Matrix<double>& band, int shift) {
    int size = triangular.nrows();

    for (ptrdiff_t i = 0; i < result.nrows() - shift; i++) {
        for (ptrdiff_t j = 0; j < size; j++) {
            if (i < size) result(i + shift, j + shift) = triangular(i, j);
            else result(i + shift, j + shift) = band(i - size, j);
        }
    }
}

#endif
