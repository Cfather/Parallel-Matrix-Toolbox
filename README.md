# Parallel-Matrix-Toolbox

Written in CUDA C++

This library enables you to perform multiple matrix operations in parallel, asynchronously.

Different problems can be set inside CUDA blocks and computed in parallel.

Number of block grids is equal to the number of different problems you need to solve. Number (size) of thread blocks is specified in comments in cuh file.

kernel <<< (Number of problems), (Parameters w.r.t size of matrix) >>> (input arguments)

You will need to change pre declared vector or matrix size in cuh file according to the scale of your problem.

Functions included in the toolbox:
- MatrixMulVector
- MatrixMulMatrix
- LinearSolver (Square matrix)
- LinearSolverMatrix (Square matrix)
- Vecnorm
- GramSmitt
