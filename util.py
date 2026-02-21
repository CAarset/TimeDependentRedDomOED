from ngsolve import H1, GridFunction, TaskManager, Integrate, Conj, LinearForm, dx
import numpy as np
from sksparse.cholmod import cholesky
from scipy.sparse import csc_matrix

#from numsa.NGSlepc import *
#import ngsolve.ngs2petsc as N2P

def ngs_to_max(ngsf, mesh, definedon = None):
    with TaskManager():
        if definedon is None:
            fes = H1(mesh, order = 0)
        else:
            fes = H1(mesh, order = 0, definedon = definedon)
        gf = GridFunction(fes)
        gf.Set(ngsf)
    return np.max(np.abs(gf.vec.FV().NumPy()[:]))

def mat_to_csc(mat, real = False):
    rows,cols,vals = mat.COO()
    if real:
        vals = vals.NumPy()[:].real
    return csc_matrix((vals, (rows, cols)), shape=(mat.height, mat.width))
    
def csc_to_chol(csc):
    factor = cholesky(csc)
    L = factor.L()
    
    chol = {}
    
    def handle_complex(f):
        def hcf(x):
            if np.any(np.iscomplex(x)):
                return f(x.real) + 1j * f(x.imag)
            return f(x.real)
        return hcf
    
    chol["apply"] = handle_complex(lambda x: csc@x)
    chol["apply_T"] = handle_complex(lambda x: csc.T@x)
    
    chol["apply_h"] = handle_complex(lambda x: L.T@factor.apply_P(x))
    chol["apply_hT"] = handle_complex(lambda x: factor.apply_Pt(L@x))
    
    chol["solve_h"] = handle_complex(lambda x: factor.apply_Pt(factor.solve_Lt(x, use_LDLt_decomposition = False)))
    chol["solve_hT"] = handle_complex(lambda x: factor.solve_L(factor.apply_P(x), use_LDLt_decomposition = False))
    
    chol["solve"] = handle_complex(lambda x: factor(x))
    chol["solve_T"] = lambda x: Exception("Not implemented.")
    
    chol["L"] = L
    chol["LT"] = L.T

    chol["hT"] = factor.apply_Pt(L)
    chol["h"] = chol["hT"].T
    
    return chol

def mat_to_eigs(mat, fes):

    U, V = fes.TnT()

    M = BilinearForm(fes)
    M += U * V * dx
    M.Assemble()

    mpre = BilinearForm(cofes)
    mpre += U * V * dx
    #mpre.mat = mpre.m + mat
    
    pre = Preconditioner(mpre, "direct", inverse="sparsecholesky")
    mpre.Assemble()

    evals, _ = solvers.PINVIT(Lap.mat, M.mat, pre = pre, num=max(2*len(omegas),12), maxit=20, printrates = False)
    return evals.NumPy()

def power_iteration(PDE, APDE, fes, iters = 10):
    u = GridFunction(fes)
    u.vec.FV().NumPy()[:] = np.random.normal(fes.ndof)
    try:
        u.vec.FV().NumPy()[:] += 1j * np.random.normal(fes.ndof)
    except:
        pass

    v = GridFunction(fes)
    w = GridFunction(fes)
    
    pair = lambda u, v: np.real(Integrate(u*Conj(u),fes.mesh))
    norm = lambda u: np.sqrt(pair(u,u))
    V = fes.TestFunction()
    
    for _ in range(iters):
        v.vec.data = PDE * LinearForm(u * V * dx).Assemble().vec
        w.vec.data = APDE * LinearForm(v * V * dx).Assemble().vec
        
        w_norm = norm(w)
        u.vec.data = w.vec.data / w_norm

    v.vec.data = PDE * LinearForm(u * V * dx).Assemble().vec
    w.vec.data = APDE * LinearForm(v * V * dx).Assemble().vec
    return pair(u, w) / pair(u, u)

def power_iteration_matrix(A, n, iters = 10):
    u = np.random.normal(size=n)
    
    pair = lambda u, v: np.real(np.inner(u,v))
    norm = lambda u: np.sqrt(pair(u,u))
    
    for _ in range(iters):
        v = A(u)
        v_norm = norm(v)
        u = v / v_norm

    return pair(u, A(u)) / pair(u, u)