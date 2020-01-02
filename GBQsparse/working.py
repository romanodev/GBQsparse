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




def cuspsolve(row,col,data,b,d):

    Acsr = sp.csr_matrix((data,(row,col)),shape=(d,d),dtype=float)

    #Acsr = sp.csr_matrix(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.empty_like(b)

    # Copy arrays to GPU

    dcsrVal = gpuarray.to_gpu(Acsr.data)
    dcsrColInd = gpuarray.to_gpu(Acsr.indices)
    dcsrIndPtr = gpuarray.to_gpu(Acsr.indptr)
    dx = gpuarray.to_gpu(x)
    db = gpuarray.to_gpu(b)

    # Create solver parameters
    n = ctypes.c_int(Acsr.shape[0])  # Need check if A is square
    m = ctypes.c_int(Acsr.shape[1])  # Need check if A is square
    nnz = ctypes.c_int(Acsr.nnz)
    nbatch = ctypes.c_int(1)
    descrA = ctypes.c_void_p()
    reorder = ctypes.c_int(0)
    tol = ctypes.c_double(1e-10)
    singularity = ctypes.c_int(0)  # -1 if A not singular
    info = ctypes.c_void_p()
    b1 = ctypes.c_int()
    b2 = ctypes.c_int()

    # create cusparse handle
    _cusp_handle = ctypes.c_void_p()
    status = _libcusparse.cusparseCreate(ctypes.byref(_cusp_handle))
    assert(status == 0)
    cusp_handle = _cusp_handle.value

    # create MatDescriptor
    status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(descrA))
    assert(status == 0)

    #create cusolver handle
    _cuso_handle = ctypes.c_void_p()
    status = _libcusolver.cusolverSpCreate(ctypes.byref(_cuso_handle))
    assert(status == 0)
    cuso_handle = _cuso_handle.value
    

    _libcusolver.cusolverSpCreateCsrqrInfo(ctypes.byref(info))
     
    _libcusolver.cusolverSpXcsrqrAnalysisBatched(cuso_handle,
                                 n,
                                 m,
                                 nnz,
                                 descrA,
                                 int(dcsrIndPtr.gpudata),
                                 int(dcsrColInd.gpudata),
                                 info)

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
    
    res = _libcusolver.cusolverSpDcsrqrsvBatched(cuso_handle,
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

    x = dx.get()  # Get result as numpy array

    # Destroy handles
    status = _libcusolver.cusolverSpDestroy(cuso_handle)
    assert(status == 0)
    status = _libcusparse.cusparseDestroy(cusp_handle)
    assert(status == 0)

    # Return result
    return x


# Test
if __name__ == '__main__':
    #n = 10

    #data = np.array([1,2,3,4,5,6],dtype=float)

    data = np.random.random_sample(6)
    b = np.random.random_sample(4)
    row = np.array([0,1,2,3,0,2],dtype=float)
    col = np.array([0,1,2,3,1,3],dtype=float)
    #b = np.array([1,1,1,1],dtype=float)
    
    #M = MSparse([0,1,2,3,0],[0,1,2,3,2],4,4)
    #M.add_LHS(np.array([[1,2,3,4,5],[1,2,3,4,5]],dtype=float))
    #:wqM.add_RHS(np.array([[1,1,1,1],[1,1,1,1]],dtype=float))

    #A = rand(n, n, density=0.7)

    x = cuspsolve(row,col,data, b,4)
    print(x)

