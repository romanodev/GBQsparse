import pycuda.gpuarray as gpuarray
import numpy as np
import ctypes
import pycuda.autoinit
from pycuda.driver import Device
import pycuda
from scipy.sparse import random
import scipy.sparse as sp
import numpy.random 
import time



# #### wrap the cuSOLVER cusolverSpDcsrqrsvBatched() using ctypes

# cuSparse
_libcusparse = ctypes.cdll.LoadLibrary('libcusparse.so')


_libcusparse.cusparseCreate.restype = int
_libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseDestroy.restype = int
_libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]




# cuSOLVER
_libcusolver = ctypes.cdll.LoadLibrary('libcusolver.so')

_libcusolver.cusolverSpCreate.restype = int
_libcusolver.cusolverSpCreate.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverSpDestroy.restype = int
_libcusolver.cusolverSpDestroy.argtypes = [ctypes.c_void_p]

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

_libcusolver.cusolverSpDcsrqrBufferInfoBatched.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p(),
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_void_p(), #info
                                            ctypes.c_void_p,
                                            ctypes.c_void_p
                                            ]


_libcusolver.cusolverSpDcsrqrsvBatched.argtypes= [ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_void_p(),
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_void_p(),
                                            ctypes.c_void_p
                                            ]
class MSparse(object):

  def __init__(self):
     
   self.init = False
   self.data = []
   self.b = []
   self.nbatch = 0

  def reset_b(self):

   self.b = []


  def add_coo(self,rows,cols,n,m):

    if self.init == False:
     self.rows = rows 
     self.cols = cols
     A = sp.csr_matrix((np.ones_like(rows), (rows, cols)), shape=(n, m))
     self.indices = A.indices
     self.indptr = A.indptr
     self.init = True
     self.nnz = A.nnz
     self.shape = A.shape
    else:
     print('Error: matrix already initiatied')


  def add_data(self,A,b):
   
    (nbatch,nd) = np.shape(A)
    if not nd == self.nnz:
       print('sparsity error')  

    self.nbatch = nbatch
    for n in range(nbatch):

     SS = sp.csr_matrix((A[n], (self.rows, self.cols)), shape=self.shape)

     self.data +=list(SS.data)
     self.b +=list(b[n])
    

  def add_csr_matrix(self,A,b):

    self.nbatch +=1
     
    if self.init == False:
      self.indices = A.indices
      self.indptr = A.indptr
      self.shape = A.shape
      self.init = True 
      self.nnz = A.nnz
    else:
      if not A.nnz == self.nnz:
        print('error')  
    
    self.data +=list(A.data)
    self.b +=list(b)

  #def to_gpu(self):
  # self.dcsrVal = gpuarray.to_gpu(self.data)
  # self.dcsrColInd = gpuarray.to_gpu(self.indices)
  # self.dcsrIndPtr = gpuarray.to_gpu(self.indptr)
  # self.nbatch = ctypes.c_int(self.nbatch)


  def solve(self):

 
   #### Prepare the matrix and parameters, copy to Device via gpuarray

   #Initialiation---
   dcsrVal = gpuarray.to_gpu(np.array(self.data))
   dcsrColInd = gpuarray.to_gpu(self.indices)
   dcsrIndPtr = gpuarray.to_gpu(self.indptr)
   nbatch = ctypes.c_int(self.nbatch)
   x = np.empty_like(np.array(self.b))
   dx = gpuarray.to_gpu(x)
   db = gpuarray.to_gpu(np.array(self.b))
   nnz = ctypes.c_int(self.nnz)
   shape = self.shape
   n = ctypes.c_int(shape[0])
   m = ctypes.c_int(shape[1])
   #-------------------------

   b1 = ctypes.c_int()
   b2 = ctypes.c_int()

   #create cusparse handle
   _cusp_handle = ctypes.c_void_p()
   status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
   #print('status: ' + str(status))
   cusp_handle = _cusp_handle.value

   #create MatDescriptor
   descrA = ctypes.c_void_p()
   status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(descrA))
   #print('status: ' + str(status))

   #create info
   info = ctypes.c_void_p()
   status = _libcusolver.cusolverSpCreateCsrqrInfo(ctypes.byref(info))
   #print('status: ' + str(status))

   #create cusolver handle
   _cuso_handle = ctypes.c_void_p()
   status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
   #print('status: ' + str(status))
   cuso_handle = _cuso_handle.value

   #print('cusp handle: ' + str(cusp_handle))
   #print('cuso handle: ' + str(cuso_handle))


   _libcusolver.cusolverSpXcsrqrAnalysisBatched(cuso_handle,
                                 n,
                                 m,
                                 nnz,
                                 descrA,
                                 int(dcsrIndPtr.gpudata),
                                 int(dcsrColInd.gpudata),
                                 info)
   t1 = time.time()

   _libcusolver.cusolverSpDcsrqrBufferInfoBatched(cuso_handle,
                           n,
                           m,
                           nnz,
                           descrA,
                           int(dcsrVal.gpudata),
                           int(dcsrIndPtr.gpudata),
                           int(dcsrColInd.gpudata),
                           nbatch,
                           info,
                           ctypes.byref(b1),
                           ctypes.byref(b2)
                           );



   w_buffer = gpuarray.zeros(b2.value, dtype=dcsrVal.dtype) 

    
   _libcusolver.cusolverSpDcsrqrsvBatched(cuso_handle,
                                 n,
                                 m,
                                 nnz,
                                 descrA,
                                 int(dcsrVal.gpudata),
                                 int(dcsrIndPtr.gpudata),
                                 int(dcsrColInd.gpudata),
                                 int(db.gpudata),
                                 int(dx.gpudata),
                                 nbatch,
                                 info,
                                 int(w_buffer.gpudata))
                                 
                      
   #print(time.time()-t1)
   # destroy handles
   status = _libcusolver.cusolverSpDestroy(cuso_handle)
   #print('status: ' + str(status))
   status = _libcusparse.cusparseDestroy(cusp_handle)
   #print('status: ' + str(status))
  
   
   return dx.reshape((self.nbatch,self.shape[0])),b2.value,cuso_handle,cusp_handle

 


if __name__ == "__main__": 

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


 x,mem = S.solve()
 print(x)
 print(mem/1024/1024)

