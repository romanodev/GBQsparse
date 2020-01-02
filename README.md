GBQsparse
========

Batched QR factorization of sparse matrices on GPUs. Wrapper of the CUDA library cusolverSpDcsrqrsvBatched()


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AQzt8f7Hy2kxgCSdqsD1nOIzH4bjK_n4)

INSTALL
========

```bash
apt-get install libsuitesparse-dev 
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
from GBQsparse import MSparse
import scipy.sparse as sp
import numpy as np
import time
import scipy.sparse.linalg as sla

N = 10
nbatch = 10
m = sp.diags([1, -2, 1], [-1, 0, 1], shape=(N, N),format='coo')

A = np.random.random_sample((nbatch,m.nnz))
B = np.random.random_sample((nbatch,N))

m = MSparse(m.row,m.col,N,nbatch,reordering=True)

m.add_LHS(A)

t1 = time.time()
X = m.solve(B)
t2 = time.time()

m.free_memory()

xs = []
for i in range(nbatch):
      S = sp.csr_matrix((A[i],(m.row,m.col)),shape=(N,N),dtype=float)
      x = sla.spsolve(S,B[i])
      xs.append(x)
xs=np.array(xs)

t3 = time.time()
print(t2-t1)
print(t3-t2)
print(np.allclose(xs,X,rtol=1e-01,atol=1e-1))
 ```
