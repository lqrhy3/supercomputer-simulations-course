#include <iostream>
#include <fstream>
#include <omp.h>
#include <cmath>
#include "array_types.hpp"
#include "array_operations.hpp"
#include "cholesky_decomposition.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    char inputFname[256]{"../chol_input_1024.txt"};

    if (argc > 1) strcpy(inputFname, argv[1]);
    ifstream stream(inputFname);

    int n = 0;
    int r = 0;
    stream >> n >> r;

    Matrix<double> mat(n, n);

    for (ptrdiff_t i = 0; i < n; i++) {
        for (ptrdiff_t j = 0; j < n; j++) {
            stream >> mat(i, j);
        }
    }

    cout << "Executing Cholesky block decomposition..." << endl;
    double start = omp_get_wtime();
    Matrix<double> factor = decompose_cholesky_block(mat, r);
    double finish = omp_get_wtime();
    cout << "Time spent: " << finish - start << '\n' << endl;

    cout << "Validating decomposition..." << endl;

    Matrix<double> result = matmul_transposed(factor, factor);

    bool isSuccessful = check_decomposition(mat, result);
    cout << "Decomposed succesfully: " << boolalpha << isSuccessful << endl;

    if (isSuccessful) {
        ofstream outputFname("chol_factor_" + to_string(n) + ".txt");
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i > j) outputFname << 0 << ' ';
                else outputFname << factor(j, i) << ' ';
            }
            outputFname << '\n';
        }
        outputFname.close();
    }

    return 0;
}
