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
#import sparseqr
import scikits.umfpack as um


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

  def __init__(self,a1,a2,d,k,reordering=False,new=False):

   self.new = False
   self.reordering = reordering
   if new:
    self.new = True
    self.dcsrIndPtr = gpuarray.to_gpu(a1)
    self.dcsrColInd = gpuarray.to_gpu(a2)
    self.n = ctypes.c_int(d) 
    self.m = ctypes.c_int(d)  
    self.nbatch = ctypes.c_int(k)
    self.nnz = ctypes.c_int(len(a2))
   else:
    #Experimental reordering
    if self.reordering:
     A = sp.coo_matrix( (np.ones(len(a2)),(a1,a2)), shape=(d,d),dtype=float)
     _, _, E, rank = sparseqr.qr(A)
     self.P = sparseqr.permutation_vector_to_matrix(E) #coo
    
     Acsr = (A*self.P.tocsr()).sorted_indices()
    else:
     Acsr = sp.csr_matrix( (np.ones(len(a2)),(a1,a2)), shape=(d,d),dtype=float)
   
    self.dcsrIndPtr = gpuarray.to_gpu(Acsr.indptr)
    self.dcsrColInd = gpuarray.to_gpu(Acsr.indices)
    self.n = ctypes.c_int(d) 
    self.m = ctypes.c_int(d)  
    self.nbatch = ctypes.c_int(k)
    self.row = a1
    self.col = a2
    self.nnz = ctypes.c_int(len(a2))

   # create cusparse handle
   self.descrA = ctypes.c_void_p()
   self.info = ctypes.c_void_p()
   _libcusolver.cusolverSpCreateCsrqrInfo(ctypes.byref(self.info))

   # create MatDescriptor
   status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(self.descrA))
   assert(status == 0)

   _cusp_handle = ctypes.c_void_p()
   status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
   assert(status == 0)
   self.cusp_handle = _cusp_handle.value

   _cuso_handle = ctypes.c_void_p()
   status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
   assert(status == 0)
   self.cuso_handle = _cuso_handle.value

   _libcusolver.cusolverSpXcsrqrAnalysisBatched(self.cuso_handle,
                                 self.n,
                                 self.m,
                                 self.nnz,
                                 self.descrA,
                                 int(self.dcsrIndPtr.gpudata),
                                 int(self.dcsrColInd.gpudata),
                                 self.info)


  def add_LHS(self,data):

   self.data = data.astype(np.float64)
   if self.new:
    self.dcsrVal = gpuarray.to_gpu(data.astype(np.float64)).ravel()
   else:
    global_data = []
    for n in range(self.nbatch.value):
     Acsr = sp.csr_matrix((data[n],(self.row,self.col)),shape=(self.n.value,self.m.value),dtype=float)
     if self.reordering:
      Acsr = (Acsr * self.P).sorted_indices()
     global_data.append(Acsr.data)
    self.dcsrVal = gpuarray.to_gpu(np.array(global_data))


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
   print(b2.value/1024/1024)

  def solve(self,b):
    #b = np.array(b)
    b = b.astype(np.float64)
    self.b = b

    #Solve scipy
    #xs = []
    #for i in range(self.nbatch.value):
      #S = sp.csr_matrix((self.data[i]/max(self.data[i]),self.dcsrColInd.get(),self.dcsrIndPtr.get()),shape=(self.n.value,self.n.value),dtype=float)
      #x = sp.linalg.spsolve(S,self.b[i]/max(self.data[i]))
    #  S = sp.csr_matrix((self.data[i],self.dcsrColInd.get(),self.dcsrIndPtr.get()),shape=(self.n.value,self.n.value),dtype=np.float32)
    #  x = sp.linalg.spsolve(S,self.b[i])
    #  xs.append(x)
    #print(np.min(xs),np.max(xs))
    #  print(min(x),max(x))
    #  print(x)
    #quit()


    transfer = False
    if isinstance(b, np.ndarray):
     b = gpuarray.to_gpu(b.astype(np.float64)).ravel()
     transfer = True 
    dx = pycuda.gpuarray.empty_like(b,np.float64) 
    

    #print(self.n) 
    #print(self.m) 
    #print(self.nbatch) 
    #print(self.nnz) 
    #print(len(self.dcsrVal)) 
    #print(len(self.dcsrColInd)) 
    #print(len(self.dcsrIndPtr)) 
    t4 = time.time()
    res = _libcusolver.cusolverSpDcsrqrsvBatched(self.cuso_handle,
                                 self.n,
                                 self.m,
                                 self.nnz,
                                 self.descrA,
                                 int(self.dcsrVal.gpudata),
                                 int(self.dcsrIndPtr.gpudata),
                                 int(self.dcsrColInd.gpudata),
                                 int(b.gpudata),
                                 int(dx.gpudata),
                                 self.nbatch,
                                 self.info,
                                 int(self.w_buffer.gpudata))
   
    # Destroy handles
    #status = _libcusolver.cusolverSpDestroy(self.cuso_handle)
    #assert(status == 0)
    #status = _libcusparse.cusparseDestroy(self.cusp_handle)
    #assert(status == 0)

    t5 = time.time()
    print(t5-t4)
    print(dx)
    #print(gpuarray.min(dx),gpuarray.max(dx))   
    xx= dx.get()
    
    aa = xx.reshape((self.nbatch.value,self.n.value))
    #print(np.min(aa,axis=1))
    #print(np.min(aa,axis=0))

    #if transfer:
    # dx = dx.get()  # Get result as numpy array

    #if self.reordering:
    # dx = np.array([self.P.dot(dx[n]) for n in range(self.nbatch.value)])


    return aa

    # Return result

  def free_memory(self):
 
    # Destroy handles
    status = _libcusolver.cusolverSpDestroy(self.cuso_handle)
    assert(status == 0)
    status = _libcusparse.cusparseDestroy(self.cusp_handle)
    assert(status == 0)



# Test
if __name__ == '__main__':

  NN = [10]
  BB = [10]
  tgpu = np.zeros((3,3))
  tcpu = np.zeros((3,3))
  for n,N in enumerate(NN):
   for b,nbatch in enumerate(BB):

    #N = 100
    #nbatch = 1000
    m = sp.diags([1, -2, 1], [-1, 0, 1], shape=(N, N),format='coo')
    A = np.random.random_sample((nbatch,m.nnz))
    B = np.random.random_sample((nbatch,N))

    #Conventional---
    #-----------------------------------
    row  = m.row
    col  = m.col
    m = MSparse(row,col,N,nbatch,reordering=False)
    m.add_LHS(A)
    X = m.solve(B)
    #t1 = time.time()
    m.free_memory()
    #-----------------------------------
    #Acsr = sp.csr_matrix((np.arange(len(m.row)),(m.row,m.col)),shape=(N,N),dtype=np.int32)
    #rot = Acsr.data
    #indptr  = Acsr.indptr
    #indices = Acsr.indices
    #m = MSparse(indptr,indices,N,nbatch,new=True)
    
    #A = np.array([ A[n,rot]  for n in range(nbatch)])
    #m.add_LHS(A)
    #X2 = m.solve(B)
    #print(np.allclose(X1,X2))
 
    #m.free_memory()

    #--------------------
    #quit()


    t2 = time.time()
    xs = []

    umfpack = um.UmfpackContext()
    xs = []
    init = False
    for i in range(nbatch):
      S = sp.csr_matrix((A[i],(m.row,m.col)),shape=(N,N),dtype=float)
      if init==False:
       umfpack.symbolic(S)
       init = True
      umfpack.numeric(S)
      x = umfpack.solve( um.UMFPACK_A,S,B[i], autoTranspose = True )
      xs.append(x)
    xs=np.array(xs)

    t3 = time.time()

    #tgpu[n,b] = t2-t1 
    #tcpu[n,b] = t3-t2
    print(X)   
    print(xs)   
    print(np.allclose(xs,X,rtol=1e-1,atol=1e-1))

  #print(tgpu)
  #print(tcpu)










