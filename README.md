GBQsparse
========

Batched QR factorization of sparse matrices on GPUs. Wrapper of the CUDA library cusolverSpDcsrqrsvBatched()


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AQzt8f7Hy2kxgCSdqsD1nOIzH4bjK_n4)

INSTALL
========

```bash
pip install GBQsparse
```

Background
==========
This package solves multiple linear systems, A x=b, sharing the same sparsity pattern.

Below are some of the resources I have used in writing GBQsparse:

[High-level discussion of Batched QR factorization](https://devblogs.nvidia.com/parallel-direct-solvers-with-cusolver-batched-qr/)

[C++ CUDA examples](https://docs.nvidia.com/cuda/cusolver/index.html)

[SKCUDA examples](https://scikit-cuda.readthedocs.io/en/latest/_modules/skcuda/cusolver.html)

[StackOverflow](https://stackoverflow.com/questions/30460074/interfacing-cusolver-sparse-using-pycuda)



Example
========

```python

 val = np.arange(1,5,dtype=np.float64)
 col = np.arange(0,4,dtype=np.int32)
 row = np.arange(0,4,dtype=np.int32)

 M = MSparse(row,col,4,4)

 M.add_LHS(np.array([[1,2,3,4],[1,2,3,4]],dtype=np.float64))
 M.add_RHS(np.array([[1,1,1,1],[1,1,1,1]],dtype=np.float64))
 x = M.solve()
 print(x)
 M.free_memory()
 ```
