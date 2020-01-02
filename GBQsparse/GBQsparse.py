# ### Interface cuSOLVER PyCUDA

from __future__ import print_function
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import scipy.sparse as sp
import ctypes
from scipy.sparse import rand
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
import time
import numpy.linalg as la
import scipy.sparse.linalg as sla
import sys
import sparseqr

## Wrap the cuSOLVER cusolverSpDcsrlsvqr() using ctypes
## http://docs.nvidia.com/cuda/cusolver/#cusolver-lt-t-gt-csrlsvqr

# cuSparse
_libcusparse = ctypes.cdll.LoadLibrary('libcusparse.so')
_libcusparse.cusparseCreate.restype = int
_libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseDestroy.restype = int
_libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseCreateMatDescr.restype = int
_libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.c_void_p]


# cuSOLVER
_libcusolver = ctypes.cdll.LoadLibrary('libcusolver.so')

_libcusolver.cusolverSpCreate.restype = int
_libcusolver.cusolverSpCreate.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverSpDestroy.restype = int
_libcusolver.cusolverSpDestroy.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverSpDcsrlsvqr.restype = int
_libcusolver.cusolverSpDcsrlsvqr.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_double,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]


_libcusolver.cusolverSpXcsrqrAnalysisBatched.restype = int
_libcusolver.cusolverSpXcsrqrAnalysisBatched.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p(),
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p() #info
                                            ]


_libcusolver.cusolverSpDcsrqrBufferInfoBatched.restype = int
_libcusolver.cusolverSpDcsrqrBufferInfoBatched.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_void_p, #info
                                            ctypes.c_void_p,
                                            ctypes.c_void_p
                                            ]



_libcusolver.cusolverSpDcsrqrsvBatched.restype = int
_libcusolver.cusolverSpDcsrqrsvBatched.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]



class MSparse(object):

  def __init__(self,row,col,d,k,reordering=False):

    #Experimental reordering
    self.reordering = reordering
    if self.reordering:
     A = sp.coo_matrix( (np.ones(len(row)),(row,col)), shape=(d,d),dtype=float)
     _, _, E, rank = sparseqr.qr(A)
     self.P = sparseqr.permutation_vector_to_matrix(E) #coo
     A = (A*self.P.tocsr()).sorted_indices()
    else:
     A = sp.csr_matrix( (np.ones(len(col)),A.indices,A.indptr), shape=(d,d),dtype=float)
   


    self.dcsrColInd = gpuarray.to_gpu(A.indices)
    self.dcsrIndPtr = gpuarray.to_gpu(A.indptr)
    self.n = ctypes.c_int(d) 
    self.m = ctypes.c_int(d)  
    self.nbatch = ctypes.c_int(k)
    self.col = col
    self.row = row
    self.nnz = ctypes.c_int(len(row))
    # Create solver parameters
    self.descrA = ctypes.c_void_p()
    self.info = ctypes.c_void_p()
    
    # create cusparse handle
    _cusp_handle = ctypes.c_void_p()
    status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
    assert(status == 0)
    self.cusp_handle = _cusp_handle.value

    # create MatDescriptor
    status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(self.descrA))
    assert(status == 0)

    #create cusolver handle
    _cuso_handle = ctypes.c_void_p()
    status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
    assert(status == 0)
    self.cuso_handle = _cuso_handle.value

    _libcusolver.cusolverSpCreateCsrqrInfo(ctypes.byref(self.info))
     
    _libcusolver.cusolverSpXcsrqrAnalysisBatched(self.cuso_handle,
                                 self.n,
                                 self.m,
                                 self.nnz,
                                 self.descrA,
                                 int(self.dcsrIndPtr.gpudata),
                                 int(self.dcsrColInd.gpudata),
                                 self.info)


  def add_LHS(self,data):

    global_data = []
    for n in range(self.nbatch.value):

     Acsr = sp.csr_matrix((data[n],(self.row,self.col)),shape=(self.n.value,self.m.value),dtype=float)
     #print(np.allclose(Acsr.todense(),Acsr.dot(self.P).todense()))
     if self.reordering:
      Acsr = (Acsr * self.P).sorted_indices()
     global_data.append(Acsr.data)

    self.dcsrVal = gpuarray.to_gpu(np.array(global_data))

    self.dx = pycuda.gpuarray.empty(self.n.value*self.nbatch.value,dtype=float) 
    self.db = pycuda.gpuarray.empty(self.n.value*self.nbatch.value,dtype=float)

    b1 = ctypes.c_int()
    b2 = ctypes.c_int()

    _libcusolver.cusolverSpDcsrqrBufferInfoBatched(self.cuso_handle,
                           self.n,
                           self.m,
                           self.nnz,
                           self.descrA,
                           int(self.dcsrVal.gpudata),
                           int(self.dcsrIndPtr.gpudata),
                           int(self.dcsrColInd.gpudata),
                           self.nbatch,
                           self.info,
                           ctypes.byref(b1),
                           ctypes.byref(b2)
                           );

    self.w_buffer = gpuarray.zeros(b2.value, dtype=self.dcsrVal.dtype) 


  def solve(self,b):

    #b = np.asarray(b, dtype=float)
    #if self.reordering:
    # self.db.set(self.P.dot(b.T).T.flatten())
    #else:
    self.db.set(b.flatten())
    
    res = _libcusolver.cusolverSpDcsrqrsvBatched(self.cuso_handle,
                                 self.n,
                                 self.m,
                                 self.nnz,
                                 self.descrA,
                                 int(self.dcsrVal.gpudata),
                                 int(self.dcsrIndPtr.gpudata),
                                 int(self.dcsrColInd.gpudata),
                                 int(self.db.gpudata),
                                 int(self.dx.gpudata),
                                 self.nbatch,
                                 self.info,
                                 int(self.w_buffer.gpudata))

    x = self.dx.get()  # Get result as numpy array
    
    x = x.reshape((self.nbatch.value,self.n.value))

    if self.reordering:
     x = np.array([self.P.dot(x[n]) for n in range(self.nbatch.value)])


    return x

    # Return result

  def free_memory(self):
 
    # Destroy handles
    status = _libcusolver.cusolverSpDestroy(self.cuso_handle)
    assert(status == 0)
    status = _libcusparse.cusparseDestroy(self.cusp_handle)
    assert(status == 0)



# Test
if __name__ == '__main__':
    

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






