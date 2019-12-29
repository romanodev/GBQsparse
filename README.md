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

#This is a class that handles multiple CSR-formetted sparse matrices with the same sparsity pattern
S = MSparse() 

#Here we prepare the master matrix, i.e. the one that dictates the sparsity pattern
N = 100
A = random(N, N, density=0.1,format='csr')
indices = A.indices
indptr = A.indptr
na = len(A.data)
 
#We create 100 random matrices and ``b'' s
nbatch = 100
data = np.random.random_sample((nbatch,na))
for n in range(nbatch):
  A = sp.csr_matrix( (data[n],indices,indptr), shape=(N,N) )
  b = np.random.random_sample((N,))
  S.add_csr_matrix(A,b) #We add the matrix and the vector at each iteration
  
#Solve the system 
x,mem = S.solve()
print(x)
print(mem/1024/1024) #We print the memory used on the device (in Mbytes)
 ```
