#ifndef MATRIX_TOOLBOX_CPP
#define MATRIX_TOOLBOX_CPP

#include "MatrixToolbox.cuh"
#include <cstdio>

__global__ void MatrixCpy(double* dst,
						  const uint32_t dst_row_num,
						  const uint32_t dst_col_num,
						  const uint32_t dst_row_start,
						  const uint32_t dst_row_end,
						  const uint32_t dst_col_start,
						  const uint32_t dst_col_end,
						  const double* src,
						  const uint32_t src_row_num,
						  const uint32_t src_col_num,
						  const uint32_t src_row_start,
						  const uint32_t src_row_end,
						  const uint32_t src_col_start,
						  const uint32_t src_col_end,
						  const double scalar,
						  bool srcTranspose){
	// use threads to parallelize
	uint32_t x_id = threadIdx.x;
	uint32_t y_id = threadIdx.y;
	
	if (x_id >= dst_row_end - dst_row_start) return;
	if (y_id >= dst_col_end - dst_col_start) return;

	uint32_t dst_id = (x_id + dst_row_start) * dst_col_num + y_id + dst_col_start;
	uint32_t src_id = 0;
	if (!srcTranspose) {
		src_id = (x_id + src_row_start) * src_col_num + y_id + src_col_start;
	}
	else {
		src_id = (y_id + src_col_start) * src_row_num + x_id + src_row_start;
	}

	if (scalar == 1.0) {
		dst[dst_id] = src[src_id];
	}
	else {
		dst[dst_id] = scalar * src[src_id];
	}

	__syncthreads();
}

__global__ void MatrixMulVector(const double* A,
								const double* x,
								double* y,
								uint32_t A_num_row,
								uint32_t A_num_col,
								bool transpose,
								double alpha,
								double beta,
						        bool ATakeAbs,
								bool xTakeAbs,									
								bool yTakeAbs) {
	// use threads to parallelize
	uint32_t t_id = threadIdx.x;

	// copy data to shared memory
	__shared__ double sha_x[MAX_PREALLOCATE_VECTOR_SIZE];
	double y_res = 0;

	if (t_id < A_num_col) {
		sha_x[t_id] = xTakeAbs ? fabs(x[t_id]) : x[t_id];
	}
	
	if (t_id < A_num_row) {
		if (beta == 0) {
			y_res = 0;
		}
		else { // compute beta * y
			y_res = beta * y[t_id];
		}
	}
	else { // we only need A_num_row threads from now
		return;
	}
	
	__syncthreads();

	// compute alpha * A * x
	if (alpha != 0) {
		if (!ATakeAbs) {
			if (!transpose) {
				for (uint32_t col = 0; col < A_num_col; col++) {
					y_res += alpha * A[t_id * A_num_col + col] * sha_x[col];
				}
			}
			else {
				for (uint32_t col = 0; col < A_num_col; col++) {
					y_res += alpha * A[col * A_num_row + t_id] * sha_x[col];
				}
			}
		}
		else {
			if (!transpose) {
				for (uint32_t col = 0; col < A_num_col; col++) {
					y_res += alpha * fabs(A[t_id * A_num_col + col]) * sha_x[col];
				}
			}
			else {
				for (uint32_t col = 0; col < A_num_col; col++) {
					y_res += alpha * fabs(A[col * A_num_row + t_id]) * sha_x[col];
				}
			}
		}
	}

	y[t_id] = yTakeAbs ? fabs(y_res) : y_res;

	__syncthreads();
}

__global__ void MatrixMulMatrix(const double* A,
								const double* x,
								double* y,
								uint32_t A_num_row,
								uint32_t A_num_col,
								uint32_t x_num_col,
								bool A_transpose,
								bool x_transpose,
								double alpha,
								double beta,
								bool ATakeAbs,
								bool xTakeAbs,
								bool yTakeAbs) {
	// use threads to parallelize
	uint32_t x_id = threadIdx.x;
	uint32_t y_id = threadIdx.y;

	// copy data to shared memory
	__shared__ double sha_A[MAX_PREALLOCATE_MATRIX_SIZE];
	__shared__ double sha_x[MAX_PREALLOCATE_MATRIX_SIZE];
	double y_res;

	if (x_id < A_num_row && y_id < A_num_col) {
		sha_A[x_id * A_num_col + y_id] = A[x_id * A_num_col + y_id];
	}

	if (x_id < A_num_col && y_id < x_num_col) {
		sha_x[x_id * x_num_col + y_id] = x[x_id * x_num_col + y_id];
	}

	if (x_id < A_num_row && y_id < x_num_col) {
		y_res = (beta == 0) ? 0 : beta * y[x_id * x_num_col + y_id];
	}
	else {
		return;
	}

	__syncthreads();

	if (alpha != 0) {
		if (!A_transpose && !x_transpose) {
			for (uint32_t i = 0; i < A_num_col; i++) {
				y_res += alpha * sha_A[x_id * A_num_col + i] * sha_x[i * x_num_col + y_id];
			}
		}
		else if (A_transpose && !x_transpose) {
			for (uint32_t i = 0; i < A_num_col; i++) {
				y_res += alpha * sha_A[i * A_num_row + x_id] * sha_x[i * x_num_col + y_id];
			}
		}
		else if (!A_transpose && x_transpose) {
			for (uint32_t i = 0; i < A_num_col; i++) {
				y_res += alpha * sha_A[x_id * A_num_col + i] * sha_x[y_id * A_num_col + i];
			}
		}
		else {
			for (uint32_t i = 0; i < A_num_col; i++) {
				y_res += alpha * sha_A[i * A_num_row + x_id] * sha_x[y_id * A_num_col + i];
			}
		}
	}

	y[x_id * x_num_col + y_id] = yTakeAbs ? fabs(y_res) : y_res;

	__syncthreads();
}

__global__ void LinearSolver(const double* A,
							 const double* b, 
							 double* sol, 
							 uint32_t n,
						     bool A_transpose,
							 bool solTakeAbs) {
	uint32_t x_id = threadIdx.x;
	uint32_t y_id = threadIdx.y;

	__shared__ double I[MAX_PREALLOCATE_MATRIX_SIZE];
	__shared__ double sha_A[MAX_PREALLOCATE_MATRIX_SIZE];
	__shared__ double sha_b[MAX_PREALLOCATE_VECTOR_SIZE];

	if (x_id < n && y_id < n) {
		I[x_id * n + y_id] = (x_id == y_id) ? 1.0 : 0.0;
		if (A_transpose) {
			sha_A[x_id * n + y_id] = A[y_id * n + x_id];
		}
		else {
			sha_A[x_id * n + y_id] = A[x_id * n + y_id];
		}

		if (y_id == 0) {
			sha_b[x_id] = b[x_id];
		}
	}
	else {
		return;
	}

	__syncthreads();

	__shared__ uint32_t newRow;
	for (uint32_t i = 0; i < n - 1; i++) {
		if (x_id == 0 && y_id == 0) {
			newRow = 0;
			if (sha_A[i * n + i] == 0) {
				for (newRow = i + 1; newRow < n; newRow++) {
					if (sha_A[newRow * n + i] != 0) break;
				}
			}
		}

		__syncthreads();

		if (i + 1 <= newRow && newRow < n) {
			if (x_id < n && y_id == 0) {
				double temp = sha_A[i * n + x_id];
				sha_A[i * n + x_id] = sha_A[newRow * n + x_id];
				sha_A[newRow * n + x_id] = temp;
			}
			else if (x_id == 0 && y_id == 1) {
				double temp = sha_b[i];
				sha_b[i] = sha_b[newRow];
				sha_b[newRow] = temp;
			}
		}

		__syncthreads();
	}

	for (uint32_t i = 0; i < n; i++) {
		if (x_id == i && x_id != y_id) {
			I[x_id*n + y_id] /= sha_A[i*n + i];
			sha_A[x_id*n + y_id] /= sha_A[i*n + i];
		}
		__syncthreads();
		if (x_id == y_id && x_id == i) {
			I[x_id*n + y_id] /= sha_A[i*n + i];
			sha_A[x_id*n + y_id] /= sha_A[i*n + i];
		}
		__syncthreads();
		if (x_id != i) {
			I[x_id*n + y_id] -= I[i*n + y_id] * sha_A[x_id*n + i];
			if (y_id != i) {
				sha_A[x_id*n + y_id] -= sha_A[i*n + y_id] * sha_A[x_id*n + i];
			}
		}
		__syncthreads();
	}

	if (y_id >= 1) return;

	// perform matrix multiplication
	double sol_res;

	if (x_id < n) {
		sol_res = 0;
	}
	else {
		return;
	}

	__syncthreads();

	for (uint32_t col = 0; col < n; col++) {
		sol_res += I[x_id * n + col] * sha_b[col];
	}

	sol[x_id] = solTakeAbs ? fabs(sol_res) : sol_res;

	__syncthreads();
}

__global__ void LinearSolverMatrix(const double* A,
								   const double* b, 
								   double* sol, 
	                               uint32_t n,
								   uint32_t b_num_col,
								   bool A_transpose,
								   bool b_transpose,
								   bool solTakeAbs){
	uint32_t x_id = threadIdx.x;
	uint32_t y_id = threadIdx.y;

	__shared__ double I[MAX_PREALLOCATE_MATRIX_SIZE];
	__shared__ double sha_A[MAX_PREALLOCATE_MATRIX_SIZE];
	__shared__ double sha_b[MAX_PREALLOCATE_MATRIX_SIZE];

	if (x_id < n && y_id < n) {
		I[x_id * n + y_id] = (x_id == y_id) ? 1.0 : 0.0;
		if (A_transpose) {
			sha_A[x_id * n + y_id] = A[y_id * n + x_id];
		}
		else {
			sha_A[x_id * n + y_id] = A[x_id * n + y_id];
		}
	}

	if (x_id < n && y_id < b_num_col) {
		if (b_transpose) {
			sha_b[x_id * b_num_col + y_id] = b[y_id * n + x_id];
		}
		else {
			sha_b[x_id * b_num_col + y_id] = b[x_id * b_num_col + y_id];
		}
	}

	__syncthreads();
	
	__shared__ uint32_t newRow;
	for(uint32_t i = 0; i < n - 1; i++){
		if (x_id == 0 && y_id == 0) {
			newRow = 0;
			if (sha_A[i * n + i] == 0) {
				for (newRow = i + 1; newRow < n; newRow++) {
					if (sha_A[newRow * n + i] != 0) break;
				}
			}
		}

		__syncthreads();

		if (i + 1 <= newRow && newRow < n) {
			if (x_id < n && y_id == 0) {
				double temp = sha_A[i * n + x_id];
				sha_A[i * n + x_id] = sha_A[newRow * n + x_id];
				sha_A[newRow * n + x_id] = temp;
			}
			else if (x_id < b_num_col && y_id == 1) {
				double temp = sha_b[i * b_num_col + x_id];
				sha_b[i * b_num_col + x_id] = sha_b[newRow * b_num_col + x_id];
				sha_b[newRow * b_num_col + x_id] = temp;
			}
		}

		__syncthreads();
	}

	for (uint32_t i = 0; i < n; i++) {
		if (x_id == i && x_id != y_id) {
			I[x_id*n + y_id] /= sha_A[i*n + i];
			sha_A[x_id*n + y_id] /= sha_A[i*n + i];
		}
		__syncthreads();
		if (x_id == y_id && x_id == i) {
			I[x_id*n + y_id] /= sha_A[i*n + i];
			sha_A[x_id*n + y_id] /= sha_A[i*n + i];
		}
		__syncthreads();
		if (x_id != i) {
			I[x_id*n + y_id] -= I[i*n + y_id] * sha_A[x_id*n + i];
			if (y_id != i) {
				sha_A[x_id*n + y_id] -= sha_A[i*n + y_id] * sha_A[x_id*n + i];
			}
		}
		__syncthreads();
	}
	
	// perform matrix multiplication
	double sol_res;

	if (x_id < n && y_id < b_num_col) {
		sol_res = 0;
	}
	else {
		return;
	}

	__syncthreads();

	for (uint32_t i = 0; i < n; i++) {
		sol_res += I[x_id * n + i] * sha_b[i * b_num_col + y_id];
	}

	sol[x_id * b_num_col + y_id] = solTakeAbs ? fabs(sol_res) : sol_res;

	__syncthreads();
}

__global__ void vecnorm(double* A, 
						double* output, 
						uint32_t rowNum, 
						uint32_t colNum) {
	uint32_t colId = threadIdx.x;
	double sum = 0;
	for (uint32_t i = 0; i < rowNum; i++) {
		double elt = A[i * colNum + colId];
		sum += elt * elt;
	}
	output[colId] = sqrt(sum);

	__syncthreads();
}

__global__ void GramSchmidt(double* A, 
							double* output, 
							uint32_t rowNum, 
							uint32_t colNum) {
	uint32_t t_id = threadIdx.x;
	__shared__ double projection[MAX_PREALLOCATE_VECTOR_SIZE];
	__shared__ double orthovec[MAX_PREALLOCATE_MATRIX_SIZE];

	for (uint32_t i = 0; i < colNum; i++) {
		if (i > 0) {
			// compute projection first
			if (t_id < i) {
				double projectionRes = 0;
				for (uint32_t j = 0; j < rowNum; j++) {
					projectionRes += output[j * colNum + t_id] * A[j * colNum + i];
				}
				projection[t_id] = projectionRes;
			}

			__syncthreads();

			// compute orthogonal vectors
			if (t_id < rowNum) {
				orthovec[t_id] = A[t_id * colNum + i];
				for (uint32_t j = 0; j < i; j++) {
					orthovec[t_id] -= projection[j] * output[t_id * colNum + j];
				}
			}
		}
		else {
			if (t_id < rowNum) {
				orthovec[t_id] = A[t_id * colNum + i];
			}
		}

		__syncthreads();

		// normalization
		__shared__ double norm;
		if (t_id == 0) {
			norm = 0;
			for (uint32_t j = 0; j < rowNum; j++) {
				double elt = orthovec[j];
				norm += elt * elt;
			}
			norm = sqrt(norm);
		}
		__syncthreads();

		if (t_id < rowNum) {
			output[t_id * colNum + i] = orthovec[t_id] / norm;
		}

		__syncthreads();
	}
}


#endif