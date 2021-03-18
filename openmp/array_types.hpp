#ifndef ARRAY_TYPES_H
#define ARRAY_TYPES_H

#include <memory>
#include <utility>
#include <cstring>

using namespace std;

template <class T>
class Vec final {
private:
    ptrdiff_t len;
    shared_ptr<T[]> data;

public:
    Vec(ptrdiff_t size) : len(size), data(new T[size]) {};
    ~Vec() = default;

    ptrdiff_t length() { return len; }
    T* raw_ptr() { return data.get(); }
    T& operator()(ptrdiff_t idx) { return data[idx]; }
};

template <class T>
class Matrix final {
private:
    ptrdiff_t rows_, columns_;
    shared_ptr<T[]> data;

public:
    Matrix(ptrdiff_t rows, ptrdiff_t columns) : rows_(rows), columns_(columns), data(new T[rows * columns]) {};
    ~Matrix() = default;

    ptrdiff_t length() { return rows_ * columns_; }
    ptrdiff_t nrows() { return rows_; }
    ptrdiff_t ncols() { return columns_; }

    T* raw_ptr() { return data.get(); }
    T& operator()(ptrdiff_t row, ptrdiff_t col) { return data[row * columns_ + col]; }
    T& operator()(ptrdiff_t idx) { return data[idx]; }

    Vec<T> row(ptrdiff_t);
    Vec<T> col(ptrdiff_t);
};

template <class T>
Vec<T> Matrix<T>::row(ptrdiff_t row_idx) {
    Vec<T> vector(columns_);
    memcpy(vector.raw_ptr(), raw_ptr() + row_idx * columns_, columns_ * sizeof(T));

    return vector;
}

template <class T>
Vec<T> Matrix<T>::col(ptrdiff_t column_idx) {
    Vec<T> vector(rows_);

    for(ptrdiff_t i = 0; i < rows_; i++) {
        vector(i) = data[i * columns_ + column_idx];
    }

    return vector;
}

#endif
