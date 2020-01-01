import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import scipy.sparse as sp
import ctypes
import sparseqr


# #### wrap the cuSOLVER cusolverSpDcsrlsvqr() using ctypes

# cuSparse
_libcusparse = ctypes.cdll.LoadLibrary('libcusparse.so')
# cuSOLVER
_libcusolver = ctypes.cdll.LoadLibrary('libcusolver.so')

_libcusparse.cusparseCreate.restype = int
_libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseDestroy.restype = int
_libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]

_libcusparse.cusparseCreateMatDescr.restype = int
_libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverSpCreate.restype = int
_libcusolver.cusolverSpCreate.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverSpDestroy.restype = int
_libcusolver.cusolverSpDestroy.argtypes = [ctypes.c_void_p]

_libcusolver.cusolverSpCreateCsrqrInfo.restype = int
_libcusolver.cusolverSpCreateCsrqrInfo.argtypes = [ctypes.c_void_p]

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
                                            ctypes.c_void_p(),
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_void_p(), #info
                                            ctypes.c_void_p,
                                            ctypes.c_void_p
                                            ]


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


_libcusolver.cusolverSpDcsrqrsvBatched.restype = int
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

_libcusparse.cusparseDestroy.restype = int
_libcusparse.cusparseDestroy.argtypes = [ctypes.c_void_p]
_libcusolver.cusolverSpDestroy.restype = int
_libcusolver.cusolverSpDestroy.argtypes = [ctypes.c_void_p]

#create cusparse handle
_cusp_handle = ctypes.c_void_p()
status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
print('status: ' + str(status))
cusp_handle = _cusp_handle.value


#create cusolver handle
_cuso_handle = ctypes.c_void_p()
status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
print('status: ' + str(status))
cuso_handle = _cuso_handle.value

print('cusp handle: ' + str(cusp_handle))
print('cuso handle: ' + str(cuso_handle))

class MSparse(object):

  def __init__(self,row,col,n,m):

    A = sp.coo_matrix( (np.ones(len(row)),(row,col)), shape=(n,m) )
    _, _, E, rank = sparseqr.qr(A)
    self.P = sparseqr.permutation_vector_to_matrix(E) #coo
    self.P = sp.eye(n)
    S = (A*self.P).tocsr()
    self.data = []
    self.b = []
    self.m = ctypes.c_int(n)
    self.n = ctypes.c_int(m)
    self.nnz = ctypes.c_int(A.nnz)
    self.dcsrColInd = gpuarray.to_gpu(S.indices)
    self.dcsrIndPtr = gpuarray.to_gpu(S.indptr)
    self.col = col
    self.row = row
    

  def add_LHS(self,A):
    
     (nbatch,_) = np.shape(A)
     data = []
     for B in A:
       C  = sp.csr_matrix((B,(self.row,self.col)), shape=(self.n.value,self.m.value) )
       C *= self.P     
       data += list(C.data)
 
     self.dcsrVal = gpuarray.to_gpu(np.array(data))
     #self.dcsrVal = gpuarray.to_gpu(A.flatten())
     self.nbatch = ctypes.c_int(nbatch)
     self.dx = pycuda.gpuarray.empty(self.n.value*self.nbatch.value,dtype=np.float64) 
     self.db = pycuda.gpuarray.empty(self.n.value*self.nbatch.value,dtype=np.float64)

     self.descrA = ctypes.c_void_p() 
     self.info = ctypes.c_void_p()
     b1 = ctypes.c_int()
     b2 = ctypes.c_int()
     status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(self.descrA))
     _libcusolver.cusolverSpCreateCsrqrInfo(ctypes.byref(self.info))
    
     _libcusolver.cusolverSpXcsrqrAnalysisBatched(cuso_handle,
                                 self.n,
                                 self.m,
                                 self.nnz,
                                 self.descrA,
                                 int(self.dcsrIndPtr.gpudata),
                                 int(self.dcsrColInd.gpudata),
                                 self.info)

     _libcusolver.cusolverSpDcsrqrBufferInfoBatched(cuso_handle,
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
 

  def add_RHS(self,B):

     self.db.set(self.P.dot(B.T).T.flatten())

  def solve(self):


    _libcusolver.cusolverSpDcsrqrsvBatched(cuso_handle,
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

    return self.dx.reshape((self.nbatch.value,self.n.value))


  def free_memory(self):
    # destroy handles
    status = _libcusolver.cusolverSpDestroy(cuso_handle)
    print('status: ' + str(status))
    status = _libcusparse.cusparseDestroy(cusp_handle)
    print('status: ' + str(status))


if __name__ == "__main__": 

 val = np.arange(1,5,dtype=np.float64)
 col = np.arange(0,4,dtype=np.int32)
 row = np.arange(0,4,dtype=np.int32)

 M = MSparse(row,col,4,4)

 M.add_LHS(np.array([[1,2,3,4],[1,2,3,4]],dtype=np.float64))
 M.add_RHS(np.array([[1,1,1,1],[1,1,1,1]],dtype=np.float64))
 x = M.solve() 
 print(x)
 M.free_memory()




