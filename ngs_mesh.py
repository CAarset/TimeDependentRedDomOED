from ngsolve import *
from xfem import *
from netgen.occ import *
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, diags
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import solve
import dill as pickle
from itertools import count, pairwise
from util import mat_to_csc, csc_to_chol # Warning: Check CHOLMOD license.
from safe_add_CFs import safe_add_CFs

from os import listdir, makedirs
from os.path import isfile, join

class mesh_maker():
    def __init__(self, PML = False,
        maxh = 0.05, refine_inner = False,
        order = 2, solver = '', domain_is_complex = False, shape = "round", a = 1, delta_t = 1e-2, rng = None,
        robin = 1, alpha = 1, scale = 1,
        **kwargs):
        
        #self.PDE = {}
        #self.APDE = {}
        self.solver = solver

        self.MassConverters = {}
        
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        
        # Prior-related parameters
        self.robin = robin
        self.alpha = alpha
        self.scale = scale

        self.delta_t = delta_t
        self.dt = lambda u: 1 / delta_t * dtref(u)
        mesh_dumps_directory = "ngs_dumps/"
        makedirs(mesh_dumps_directory, exist_ok = True)
        
        target_filename = "maxh_" + str(round(maxh,3)) + \
                          "_order_" + str(order) + "_refine_" + str(1*refine_inner) + "_dt_" + str(round(self.delta_t,5)) + "_robin_" + str(round(self.robin,5)) + "_alpha_" + str(round(self.alpha,5)) + "_scale_" + str(round(self.scale,5))
        self.target_filename = target_filename
        
        load_flag = False
        for filename in listdir(mesh_dumps_directory):
            if isfile(join(mesh_dumps_directory, filename)):
                if target_filename in filename:
                    with open(mesh_dumps_directory + filename, "rb") as input_file:
                        obj = pickle.load(input_file)

                        self.mesh = obj["mesh"]
                        self.designmesh = obj["designmesh"]

                        self.fes = obj["fes"]
                        self.cofes = obj["cofes"]
                        
                        self.M = obj["M"]
                        self.coM = obj["coM"]
                        
                        self.diagPrior = obj["diagPrior"]
                        self.tracePrior = obj["tracePrior"]
                        
                        load_flag = True
                        break
            
        if not load_flag:
            print("No saved mesh found, generating...")
            filename = mesh_dumps_directory + target_filename

            # Create mesh
            with TaskManager():
                
                air = MoveTo(-1,-1).Rectangle(2,2).Face()
                
                scat = Circle((-0.2, -0.6), 0.05).Face() + Circle((0.6, -0.4), 0.1).Face() + Circle((-0, 0.3), 0.3).Face()
                scat.edges.name = "scat_edges"
                
                air = air - scat
                air.faces.name = "air"
                air.edges.name = "outer_edges"
                
                src = MoveTo(-1, -1).Rectangle(0.5, 2).Face()
                src.faces.name = "src"
                src.edges.name = "src_edges"
                
                outer = air - src
                
                outer.edges.Max(X).name = "right"
                (outer.edges.Max(Y)+src.edges.Max(Y)).name = "top"
                (outer.edges.Min(Y)+src.edges.Min(Y)).name = "bottom"
                outer.edges.Min(X).name = "bad"
                src.edges.Min(X).name = "left"
                scat.edges.name = "scat_edges"
                
                outer.faces.name = "outer"                
                geo = Glue([src, outer])
                mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh = maxh))

                designmesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=0.01))
                #designmesh = deepcopy(mesh)
                #for el in mesh.Elements():
                #    designmesh.SetRefinementFlag(el, el.mat == "outer_face")
                #designmesh.Refine()

                if refine_inner:
                    for _ in range(int(refine_inner)):
                        # Refine inner part to emphasise source reconstruction.
                        for el in mesh.Elements():
                            mesh.SetRefinementFlag(el, el.mat == "src")
                        mesh.Refine()
                    print("Doing post-refine...")
                    for el in mesh.Elements():
                        mesh.SetRefinementFlag(el, True)
                    mesh.Refine()
                    
                
                self.mesh = mesh
                self.designmesh = designmesh
                
                # Fes creation
                
                self.fes = H1(self.mesh, order = order, definedon = "src")
                self.fes = Compress(self.fes)
                
                self.cofes = H1(self.mesh, order = order)#, dirichlet="right|scat_edges")
                self.cofes = Compress(self.cofes)
                
        # Don't save these
        self.fes0 = L2(self.mesh, order = 0, definedon = "src")
        self.fes0 = Compress(self.fes0)
        self.designfes = self.fes#L2(self.designmesh, order = order, definedon = "src")
        #self.designfes = Compress(self.designfes)
        self.codesignfes = self.cofes#L2(self.designmesh, order = order)

        self.tfes = ScalarTimeFE(order = 2)
        self.tcofes = self.tfes * self.cofes

        # Create mass matrices etc. for all these FESes
        for fes in [self.fes, self.cofes]:
            self.create_Mass(fes)

        # n is the number of free source dofs.
        self.n = self.fes.ndof
        if not load_flag:
            print("Finished meshing (n = " + str(self.n) + ", n_cofes = " + str(self.cofes.ndof) + ").")
        # Trial-test pairs for the two feses.

        u, v = self.fes.TnT()
        U, V = self.cofes.TnT()
        
        # The linear operators we require are mass matrices,
        # prior covariance (half power), PDE forward and PDE adjoint.

        self.Mass = BilinearForm(self.fes)
        self.Mass += u * v * dx
        self.Mass.Assemble()
        self.Mass = self.Mass.mat
        
        self.coMass = BilinearForm(self.cofes)
        self.coMass += U * V * dx
        self.coMass.Assemble()
        self.coMass = self.coMass.mat

        # Bilaplacian as prior
        # Use Robin boundary condition to avoid boundary effects

        beta = 1
        cor_length = 0.3
        alpha = self.alpha #1/25
        
        scale = self.scale
        robin = np.sqrt(alpha * beta)/1.42/scale
        robin *= self.robin
        
        print("cor_length:", cor_length, "vs. domain radius", 0.5, "alpha:", alpha, ", scale:", scale)
        
        self.PriorInv = BilinearForm(self.fes)
        self.PriorInv += 1/scale * alpha * grad(u) * grad(v) * dx + 1/scale * beta * u * v * dx
        self.PriorInv += robin * u * v * ds
        self.PriorInv.Assemble()
        self.PriorInv = self.PriorInv.mat
        self.Prior = self.PriorInv.Inverse(freedofs = self.fes.FreeDofs(), inverse=solver)

        # PDE creation

        self.a = a
        self.make_PDE(a = self.a)
        self.u = GridFunction(self.cofes)

        if load_flag:
            self.Mchol = csc_to_chol(self.M)
            self.Kinv = lambda X: spsolve(self.K,X)
            self.coMinv = lambda X: spsolve(self.coM,X)
        else:
            self.M = mat_to_csc(self.Mass, real = True)
            self.Mchol = csc_to_chol(self.M)

            self.coM = mat_to_csc(self.coMass, real = True)
            
            # Assemble prior covariance field and trace
            
            PriorCov = lambda f: self.C(self.C(f))
            self.diagPrior = self.diag(Cov = self.C, power = 2)
            self.tracePrior = self.trace(PriorCov)
            print("Prior trace: ",self.tracePrior,"...",sep="")
            
            # Save output        
            obj = {"mesh": self.mesh, "designmesh": self.designmesh, \
                   "fes": self.fes, "cofes": self.cofes, \
                   "M": self.M, "coM": self.coM, \
                   "diagPrior": self.diagPrior, "tracePrior": self.tracePrior}
            with open(filename, "wb") as output_file:
                pickle.dump(obj, output_file)    
            print("Successfully stored mesh.")
        
    def create_Mass(self, fes1, fes2 = None):
        # Builds mass matrix and accompanying Cholesky 
        # decomposition if not already present
        # Can also build the converter between feses

        # Must decide if we need to build this
        if fes2 is None:
            fes2 = fes1

        # Workaround needed for complex-to-real conversion:
        # Map from real to complex and transpose.
        complex_to_real = fes1.is_complex and not fes2.is_complex

        def create(fes1, fes2, create_Mchol = True):
            u1 = fes1.TrialFunction()
            v2 = fes2.TestFunction()

            with TaskManager():
                MassMatrix = BilinearForm(trialspace=fes1, testspace=fes2)
                MassMatrix += u1 * v2 * dx
                MassMatrix.Assemble()
                MassMatrix = MassMatrix.mat

                if create_Mchol:
                    MassMatrixCSC = mat_to_csc(MassMatrix)
                    # For 0th order FEM, we throw away off-diagonals
                    if fes1.globalorder == 0:
                        MassMatrixCSC = diags(MassMatrixCSC.diagonal(), format = "csc")
                    Mchol = csc_to_chol(MassMatrixCSC)
                    return MassMatrix, Mchol
                else:
                    return mat_to_csc(MassMatrix)

        def inner(f,g):
            assert f.space == g.space, "Spaces do not match!"
            space = f.space
            Mg = GridFunction(space)
            Mg.vec.data = space.MassMatrix * g.vec
            return np.vdot(Mg.vec.FV().NumPy()[:], f.vec.FV().NumPy()[:]).real

        def norm(f):
            return inner(f,f)**(1/2)

        # Populate both feses with mass matrix and Cholesky decomp
        for fes in [fes1, fes2]:
            if not hasattr(fes, "MassMatrix"):
                fes.MassMatrix, fes.Mchol = create(fes1 = fes, fes2 = fes, create_Mchol = True)
                fes.MassMatrixCSC = mat_to_csc(fes.MassMatrix)
                fes.MassMatrixDiagonal = fes.MassMatrixCSC.diagonal()
                fes.inner = inner
                fes.norm = norm
            
        if fes1 != fes2:
            if id(fes1) not in self.MassConverters.keys():
                self.MassConverters[id(fes1)] = {}
            if id(fes2) not in self.MassConverters.keys():
                self.MassConverters[id(fes2)] = {}
                
            if id(fes2) not in self.MassConverters[id(fes1)].keys():
                #try:
                #    # More efficient to just use the Hermitian as the opposite-direction mass matrix
                #    self.MassConverters[id(fes1)][id(fes2)] = self.MassConverters[id(fes2)][id(fes1)].T
                #except:
                self.MassConverters[id(fes1)][id(fes2)] = create(fes1 = fes1, fes2 = fes2, create_Mchol = False)
                        
    def fes_to_fes(self, f, fes2):
        
        assert hasattr(f, "space"), "f must have a space to convert from!"
        fes1 = f.space
        
        if fes1 == fes2:
            print("Input and output feses are equal, check if this code needs to be called here...")
            return f
        else:
            self.create_Mass(fes1 = fes1, fes2 = fes2)
            partially_interpolated = GridFunction(fes2)
            interpolated = GridFunction(fes2)
            
            partially_interpolated.vec.FV().NumPy()[:] = self.MassConverters[id(fes1)][id(fes2)] @ f.vec.FV().NumPy()[:]
            interpolated.vec.data = fes2.MassMatrix.Inverse(freedofs = fes2.FreeDofs()) * partially_interpolated.vec
            return interpolated
                
    def coeff_to_ngs(self, r, fes = None):
        
        if fes is None:
            fes = self.fes
            
        with TaskManager():
            f = GridFunction(fes)
            if fes.is_complex:
                f.vec.FV().NumPy()[:] = r
            else:
                f.vec.FV().NumPy()[:] = r.real
        return f
        
    def ngs_to_coeff(self, f, fes = None):
        if fes is None:
            try:
                fes = f.space
            except:
                fes = self.fes
        
        # Allows sending CFs to coeffs
        if not hasattr(f, "space"):
            fs = GridFunction(fes)
            fs.Set(f)
            f = fs
        
        with TaskManager():
            if fes.is_complex:
                return f.vec.FV().NumPy()[:].conj()
            else:
                return f.vec.FV().NumPy()[:].real

    def real_to_ngs(self, r, fes = None, full_power = False):
        if fes is None:
            fes = self.fes
        self.create_Mass(fes)
        solve = fes.Mchol["solve_h"](r)
        if full_power:
            solve = fes.Mchol["solve_hT"](solve)
        return self.coeff_to_ngs(solve, fes = fes)

    def ngs_to_real(self, f, fes = None, full_power = False):
        if fes is None:
            try:
                fes = f.space
            except:
                fes = self.fes
        self.create_Mass(fes)
        if not full_power:
            apply = fes.Mchol["apply_h"]
        else:
            apply = fes.Mchol["apply"]
        return apply(self.ngs_to_coeff(f, fes = fes))
           
    def make_PDE(self, a):
        
        self.dxt = self.delta_t * dxtref(self.mesh, time_order=2)
        self.dxold = dmesh(self.mesh, tref=0)
        self.dxnew = dmesh(self.mesh, tref=1)
        
        # Heat PDE
        # Neumann boundary condition on scatterers implicitly imposed by not including it (=0)
        
        with TaskManager():
            Ut, Vt = self.tcofes.TnT()

            PDE = BilinearForm(self.tcofes, symmetric=False)
            PDE += (self.dt(Ut) * Vt + a * grad(Ut) * grad(Vt)) * self.dxt
            PDE += Ut * Vt * self.dxold
            PDE.Assemble()
            PDE = PDE.mat.Inverse(freedofs=self.tcofes.FreeDofs())
            
        self.PDE = PDE
        return
        
    def TimeStepping(self, source = 0, initial_cond = 0, t0 = 0, t1 = 1, integrate = False, save = False):
        
        tcofes = self.tcofes
                
        gfu = GridFunction(tcofes)
        u_last = CreateTimeRestrictedGF(gfu, 1)
        if not initial_cond == 0:
            u_last.vec.FV().NumPy()[:] = 1*initial_cond.vec.FV().NumPy()[:]
        
        u_out = GridFunction(u_last.space)
        if integrate:
            u_out.vec.data += self.delta_t / 2 * u_last.vec.data
            
        t = Parameter(0)
        t.Set(t0)

        if save:
            gfut = GridFunction(u_last.space, multidim=0)

        _, Vt = self.tcofes.TnT()
        
        Lf = LinearForm(self.tcofes)
        Lf += source * Vt * self.dxt
        Lf += u_last * Vt * self.dxold
        
        while t1 - t.Get() > self.delta_t / 2:
            with TaskManager():
                if save:
                    gfut.AddMultiDimComponent(u_last.vec)
                Lf.Assemble()
                gfu.vec.data = self.PDE * Lf.vec
                RestrictGFInTime(spacetime_gf=gfu, reference_time=1.0, space_gf=u_last)
                t.Set(t.Get() + self.delta_t)
                    
                if integrate:
                    if t1 - t.Get() > self.delta_t / 2:
                        u_out.vec.data += self.delta_t * u_last.vec.data
                    else:
                        u_out.vec.data += self.delta_t / 2 * u_last.vec.data
                else:
                    u_out.vec.data = u_last.vec.data
            print("\rt = {0:12.9f}".format(t.Get()), end="")
        if save:
            gfut.AddMultiDimComponent(u_last.vec)
            return u_out, gfut
        else:
            return u_out
    
    def A(self, f, t0 = 0, t1 = 1):

        u = self.TimeStepping(source = f, t0 = t0, t1 = t1, save = False)
        return u

    def AT(self, u, t0 = 0, t1 = 1):

        F = self.TimeStepping(initial_cond = u, t0 = t0, t1 = t1, integrate = True, save = False)
        #f = GridFunction(self.fes)
        #f.Set(F)
        return self.fes_to_fes(F, fes2 = self.fes)
        
    def C(self, f):
        with TaskManager():
            v = self.fes.TestFunction()
            F = GridFunction(self.fes)
            F.vec.data = self.Prior * LinearForm(f * v * dx).Assemble().vec
        return F

    def CT(self, f):
        return self.C(f)
    
    # Diagonal by 0th-order approximation
    def field(self, Cov, fes = None, squared = False):
        
        if fes is None:
            fes = self.fes
        fes0 = self.fes0

        dia = np.empty(fes0.ndof)
        self.create_Mass(fes0)
        diagM0 = fes0.MassMatrixDiagonal
        inds0 = np.flatnonzero(diagM0)
        for i in inds0:
            print("\r","Computing pointwise variance field index ",i,"/",len(inds0),"...",sep="",end="")
            e = np.zeros(fes0.ndof)
            e[i] = 1
            e = self.coeff_to_ngs(e, fes = fes0)
            e = Cov(e)
            if squared:
                e = Cov(e)
            if e.space is not fes0:
                self.create_Mass(fes0,e.space)
                e = self.fes_to_fes(e, fes2 = fes0)
            e = self.ngs_to_coeff(e)
            dia[i] = e[i]
        dia[inds0] /= diagM0[inds0]
        dia = self.coeff_to_ngs(dia, fes = fes0)
        return dia
    
    def diag(self, Cov, fes = None, order = None, power = 1, interp = False):

        real_to_ngs = self.real_to_ngs
        ngs_to_real = self.ngs_to_real
        
        if fes is None:
            try:
                fes = Cov(CF(1)).space
            except:
                fes = self.fes

        n = fes.ndof
        def realCov(r):
            f = real_to_ngs(r,fes=fes)
            f = Cov(f)
            return ngs_to_real(f)

        CovOp = LinearOperator(matvec = realCov, rmatvec = realCov, shape = (n,n))
        
        if order is None:
            from scipy.linalg.interpolative import estimate_rank
            from time import time
            start = time()
            order = estimate_rank(A = CovOp, eps = 1e-15**(1/power))
            print("Estimated rank in ",round(time()-start,2),"s...",sep="")
            print("Estimated numerical rank of Cov:",order)
            #order = n-1

        while order > 0:
            try:
                if isinstance(Cov,la.BaseMatrix):

                    def input_basevec(basevec):
                        f = GridFunction(fes)
                        f.vec[:] = basevec
                        return f
                        
                    eigvals, eigvecs = solvers.LOBPCG(mata = Cov, matm = fes.MassMatrix, 
                                                    pre = Cov.Inverse(freedofs = fes.FreeDofs(), inverse = self.solver), 
                                                    num = order, maxit = self.max_eig_iters, printrates=True)
                    diag_generator = (input_basevector(eigvec)**2 for eigvec in eigvecs)
                
                else:
                    
                    eigvals, eigvecs = eigsh(CovOp, k = min(order,n-1))
                    diag_generator = (real_to_ngs(eigvec)**2 for eigvec in eigvecs.T)
                print("Biggest eigval: ",max(eigvals),", smallest: ",min(eigvals),", ratio: ",min(eigvals)/max(eigvals),"...",sep="")
                diagCF = safe_add_CFs(CFs = diag_generator, weights = iter(eigvals**power), length = order)
                if interp:
                    diag = GridFunction(fes)
                    diag.Set(diagCF)
                    return diag
                else:
                    return diagCF
            except:
                order -= 1
                print("\rReducing order to ",order," and retrying diag computation...",sep="",end="")
        raise Exception("Diag computation failed.")

    def trace(self, Cov, fes = None):
        # Computes the trace of any arbitrary
        # ngsolve operator
        coeff_to_ngs = self.coeff_to_ngs
        
        if fes is None:
            try:
                fes = Cov(CF(1)).space
            except:
                fes = self.fes
        
        n = self.n
        tr = 0

        # Unit vectors (Euclidean)
        def e(i):
            e = np.zeros(n)
            e[i] = 1
            return e

        for i in range(n):
            print("\rComputing cov trace, index " + str(i) + "/" + str(n) + "...", sep="",end="")
            tr += self.ngs_to_real(Cov(self.real_to_ngs(e(i))))[i]
        print("")
        return tr
    
    def sample(self, Ch):
        eta = self.rng.normal(size=self.n)
        eta = self.real_to_ngs(eta)
        sample = Ch(eta)
        return sample
