#ifndef MATRIX_TOOLBOX_H
#define MATRIX_TOOLBOX_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>
#define MAX_PREALLOCATE_VECTOR_SIZE 32
#define MAX_PREALLOCATE_MATRIX_SIZE 1024

/*
Matrix operation with vector

y = yTakeAbs(alpha * ATakeAbs(A) * xTakeAbs(x) + beta * y)

dim(x) == A_num_col
dim(y) == dim(b) == A_num_row

A is majored in row by default

1 2 ==> 1 2 3 4
3 4

if transpose is true, then A_num_row and A_num_col should be
the number of rows and cols of A^T

if ~TakeAbs is true, the absolute value is computed

<<< 1, max(A_num_row, A_num_col) >>>
*/
__device__ void MatrixMulVector(const double* A,
								const double* x,
								double* y,
								uint32_t A_num_row,
								uint32_t A_num_col,
								bool transpose = false,
								double alpha = 1.0,
								double beta = 0.0,
						        bool ATakeAbs = false,
								bool xTakeAbs = false,											
								bool yTakeAbs = false);

/*
Matrix operation with another matrix

y = yTakeAbs(alpha * ATakeAbs(A) * xTakeAbs(x) + beta * y)

x_num_row == A_num_col

A is majored in row by default

1 2 ==> 1 2 3 4
3 4

if A_transpose is true, then A_num_row and A_num_col should be
the number of rows and cols of A^T

if x_transpose is true, then and x_num_col should be
the number of cols of x^T

if ~TakeAbs is true, the absolute value is computed

<<< 1, (max(A_num_row, A_num_col), max(x_num_row, A_num_col)) >>>
*/
__device__ void MatrixMulMatrix(const double* A,
								const double* x,
								double* y,
								uint32_t A_num_row,
								uint32_t A_num_col,
								uint32_t x_num_col,
								bool A_transpose = false,
								bool x_transpose = false,
								double alpha = 1.0,
								double beta = 0.0,
								bool ATakeAbs = false,
								bool xTakeAbs = false,
								bool yTakeAbs = false);

/*
Linear solver
A * sol = b
where A is square with dim == n

A is stored in row

1 2 ==> 1 2 3 4
3 4

<<< 1, (n, n) >>>
*/
__device__ void LinearSolver(const double* A,
							 const double* b,
							 double* sol,
							 uint32_t n,
							 bool A_transpose = false,
							 bool solTakeAbs = false);

/*
Linear solver
A * sol = b
where A is square with dim == n and b is a matrix rather than 
a vector

A is stored in row

1 2 ==> 1 2 3 4
3 4

<<< 1, (n, max(n, b_num_col)) >>>
*/
__device__ void LinearSolverMatrix(const double* A,
								   const double* b, 
								   double* sol, 
	                               uint32_t n,
								   uint32_t b_num_col,
								   bool A_transpose = false,
								   bool b_transpose = false,
								   bool solTakeAbs = false);

/*
column vector 2-norm
*/
__device__ void vecnorm(double* A,
						double* output, 
						uint32_t rowNum, 
						uint32_t colNum);

/*
Perform Gram-Schmidt process to all the column vectors of A
*/
__device__ void GramSchmidt(double* A,
							double* output,
							uint32_t rowNum,
							uint32_t colNum);

#endif