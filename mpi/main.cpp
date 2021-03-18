#include <iostream>
#include <fstream>
#include <omp.h>
#include <cmath>
#include "mpi.h"

#include "array_types.hpp"
#include "array_operations.hpp"
#include "cholesky_decomposition.hpp"

using namespace std;

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int myrank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    char input_file[256]{"../chol_input_1024x1024.txt"};

    if (argc > 1) {
        strcpy(input_file, argv[1]);
    }
    ifstream stream(input_file);

    ptrdiff_t size = 0;
    int block_size = 0;

    stream >> size >> block_size;

    Matrix<double> original(size, size);

    for (ptrdiff_t row_iter = 0; row_iter < size; row_iter++) {
        for (ptrdiff_t column_iter = 0; column_iter < size; column_iter++) {
            stream >> original(row_iter, column_iter);
        }
    }

     if (myrank == 0)
         cerr << "Decomposing... /" << comm_size << " process(es) are used/" << endl;
    double starttime, endtime;
    starttime = MPI_Wtime();

    Matrix<double> triangular = decompose_cholesky_block(original, MPI_COMM_WORLD, comm_size, 0, block_size);

    endtime = MPI_Wtime();
     if (myrank == 0) {
         cerr << "Time spent: " << endtime - starttime << endl;
         cerr << "Validating decomposition..." << endl;
     }

     Matrix<double> result(triangular.nrows(), triangular.nrows());
     matmul_transposed_mpi(triangular.raw_ptr(), triangular.raw_ptr(), result.raw_ptr(),
                         triangular.nrows(), triangular.ncols(), triangular.nrows(), MPI_COMM_WORLD, comm_size, 0);


     if (myrank == 0) {
         bool is_successful = check_decomposition(original, result);
         cerr << "Decomposed succesfully: " << boolalpha << is_successful << endl;
     }

    if (myrank == 0) {
//         ofstream output_fname("chol_factor_" + to_string(size) + ".txt");
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                if (j < i) cout << 0 << ' ';
                else cout << triangular(j, i) << ' ';
            }
            cout << '\n';
        }
//         output_fname.close();
    }

    MPI_Finalize();

    return 0;
}
