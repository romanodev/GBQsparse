GBQsparse
========

Batched QR factorization of sparse matrices on GPUs. Wrapper of the CUDA library cusolverSpDcsrqrsvBatched()


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vxz6taqxNGwhbRNKO-uyEb4y6HZhZhHI)

INSTALL
========

```bash
pip install GBQsparse
```

Example
========

```python
 N = 100
 A = random(N, N, density=0.1,format='csr')

 indices = A.indices
 indptr = A.indptr
 na = len(A.data)
 nbatch = 100
 S = MSparse()
 data = np.random.random_sample((nbatch,na))
 for n in range(nbatch):
  A = sp.csr_matrix( (data[n],indices,indptr), shape=(N,N) )
  b = np.random.random_sample((N,))
  S.add_csr_matrix(A,b)


 x,mem = S.inv()
 print(x)
 print(mem/1024/1024)
 ```
