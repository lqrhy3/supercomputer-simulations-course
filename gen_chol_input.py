import numpy as np
from numpy.linalg import matrix_rank
import sys

if __name__ == '__main__':
    # Generates symmetric positive definite matrix with given size
    try:
        # matrix size
        n = int(sys.argv[1])
    except IndexError:
        n = 1024

    try:
        # cholesky decomposition block size
        r = int(sys.argv[2])
    except IndexError:
        r = n // 16

    while True:
        mat = np.random.uniform(-4000, 4000, size=(n, n))
        if matrix_rank(mat) == n:
            break

    mat = np.dot(mat, mat.T)

    with open('chol_input_{0}.txt'.format(n), 'wb+') as f:
        # f.write(bytes(str(n) + '\n' + str(r) + '\n')) # Python 2.x
        f.write(bytes(str(n) + '\n' + str(r) + '\n', encoding='utf-8')) # Python 3.x
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')
