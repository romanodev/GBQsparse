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
import scikits.umfpack as um
from GBQsparse import MSparse
from matplotlib.pylab import *
from openbte.fig_maker import *

fonts = init_plotting()

if __name__ == '__main__':

  NN = [10]
  BB = [10,100,1000]
  tgpu = np.zeros((3,3))
  tcpu = np.zeros((3,3))
  for n,N in enumerate(NN):
   for b,nbatch in enumerate(BB):

    #N = 100
    #nbatch = 1000
    m = sp.diags([1, -2, 1], [-1, 0, 1], shape=(N, N),format='coo')

    A = np.random.random_sample((nbatch,m.nnz))
    B = np.random.random_sample((nbatch,N))

    m = MSparse(m.row,m.col,N,nbatch,reordering=False)

   
    m.add_LHS(A)
    t1 = time.time()
    X = m.solve(B)

    m.free_memory()

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

    tgpu[n,b] = t2-t1 
    tcpu[n,b] = t3-t2
    #print(t2-t1)
    #print(t3-t2)
    
    print(np.allclose(xs,X,rtol=1e-01,atol=1e-1))

  print(tgpu)
  quit()
  plot(BB,tcpu[0],marker='o',color=c1)
  plot(BB,tgpu[0],marker='o',color=c1,ls='--')
  plot(BB,tcpu[1],marker='o',color=c2)
  plot(BB,tgpu[1],marker='o',color=c2,ls='--')
  plot(BB,tcpu[2],marker='o',color=c3)
  plot(BB,tgpu[2],marker='o',color=c3,ls='--')
  xscale('log')
  yscale('log')
  xlabel('Batch size',fontproperties=fonts['regular'])
  ylabel('Time',fontproperties=fonts['regular'])
  xticks(fontproperties=fonts['regular'])
  yticks(fontproperties=fonts['regular'])
  f = fonts['regular']
  f.set_size(20)
  legend(['CPU [N=10]','GPU [N=10]','CPU [N=100]','GPU [N=100]','CPU [N=1000]','GPU [N=1000]'],prop=f,ncol=2)
  ylim([1e-4,5e2])

  show()
  









