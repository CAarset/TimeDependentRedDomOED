from ngsolve import *
from xfem import *
from netgen.occ import *
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
import dill as pickle
from itertools import count, pairwise
from util import mat_to_csc, csc_to_chol # Warning: Check CHOLMOD license.

from os import listdir, makedirs
from os.path import isfile, join

class mesh_maker():
    def __init__(self, PML = False, \
        maxh = 0.05, refine_inner = False, \
        order = 3, solver = '', domain_is_complex = False, shape = "round", a = 1, delta_t = 1e-2, rng = None, \
        robin = 1, alpha = 1, scale = 8.5, **kwargs):
        
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

                        #self.fes = obj["fes"]
                        #self.cofes = obj["cofes"]
                        #self.fes0 = obj["fes0"]
                        
                        self.M = obj["M"]
                        self.coM = obj["coM"]
                        
                        self.K = obj["K"]
                        
                        self.fieldPrior = obj["fieldPrior"]
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
                
                self.mesh = mesh
                self.designmesh = designmesh
                
        # Fes creation

        self.fes = H1(self.mesh, order = order, definedon = "src")#, dirichlet = "top|left|bottom")
        self.fes = Compress(self.fes)
        
        self.fes0 = L2(self.mesh, order = 0, definedon = "src")
        self.fes0 = Compress(self.fes0)
        
        self.cofes = H1(self.mesh, order = 2, dirichlet="right|scat_edges")#|top|bottom|left")
        self.cofes = Compress(self.cofes)
        
        # Don't save these
        self.designfes = L2(self.designmesh, order = order + 2, definedon = "src")
        self.codesignfes = L2(self.designmesh, order = order + 2)

        self.tfes = ScalarTimeFE(order = 2)
        self.tcofes = self.tfes * self.cofes

        # gfut to store space-time-fes data
        # gfu to store space-fes data
        self.gfut = GridFunction(self.tcofes)
        #self.gfu = CreateTimeRestrictedGF(self.gfut, 1)
        self.gfu = GridFunction(self.fes)

        # Create mass matrices etc. for all these FESes
        for fes in [self.fes, self.fes0, self.cofes]:
            self.create_Mass(fes)

        # n is the number of free source dofs.
        self.n = self.fes.ndof
        if not load_flag:
            print("Finished meshing (n = " + str(self.n) + ", n_cofes = " + str(self.cofes.ndof) + ").")
        # Trial-test pairs for the two feses.

        u, v = self.fes.TnT()
        u0, v0 = self.fes0.TnT()
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

        # Redo of the mass matrix for the L2-0th order FES
        # This is diagonal, so we will throw away everything else        
        self.Mass0 = BilinearForm(self.fes0)
        self.Mass0 += u0 * v0 * dx
        self.Mass0.Assemble()
        self.Mass0 = self.Mass0.mat
        self.diagM0 = mat_to_csc(self.Mass0).diagonal()

        # Bilaplacian as prior
        # Use Robin boundary condition to avoid boundary effects

        beta = 1
        cor_length = 0.3
        alpha = self.alpha #1/25
        
        scale = self.scale
        robin = np.sqrt(scale**2 * alpha * beta)/1.42
        robin *= self.robin
        
        print("cor_length:", cor_length, "vs. domain radius", 0.5, "alpha:", alpha, ", scale:", scale)
        
        self.PriorInv = BilinearForm(self.fes)
        self.PriorInv += scale * alpha * grad(u) * grad(v) * dx + scale * beta * u * v * dx
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
            self.coMinv = lambda X: spsolve(self.coM,X)
            
            self.K = mat_to_csc(self.PriorInv)
            self.Kinv = lambda X: spsolve(self.K,X)
            
            # Assemble prior covariance field and trace
            
            PriorCov = lambda f: self.C(self.C(f))
            self.fieldPrior = self.cov_to_field(PriorCov)
            self.tracePrior = self.cov_to_trace(PriorCov)
            print("Prior trace: ",self.tracePrior,"...",sep="")
            
            # Save output        
            obj = {"mesh": self.mesh, "designmesh": self.designmesh, \
                   "fes": self.fes, "cofes": self.cofes, "fes0": self.fes0, \
                   "M": self.M, "coM": self.coM, \
                   "K": self.K, \
                   "fieldPrior": self.fieldPrior, "tracePrior": self.tracePrior}
            with open(filename, "wb") as output_file:
                pickle.dump(obj, output_file)    
            print("Successfully stored mesh.")
        
        self.KMhT = lambda x: self.Kinv(self.Mchol["apply_hT"](x))
    
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
            PDE = PDE.mat.Inverse(self.tcofes.FreeDofs())
            
        self.PDE = PDE        
        return
        
    def TimeStepping(self, source = 0, initial_cond = 0, t0 = 0, t1 = 1, integrate = False, save = False):
        
        gfut = self.gfut
        u_last = self.gfu
        u_last.Set(initial_cond)
        
        u_out = GridFunction(u_last.space)
        if integrate:
            u_out.vec.data += self.delta_t / 2 * u_last.vec.data
            
        t = Parameter(0)
        t.Set(t0)

        if save:
            gfut_out = GridFunction(u_last.space, multidim=0)

        _, Vt = self.tcofes.TnT()
        
        Lf = LinearForm(self.tcofes)
        Lf += source * Vt * self.dxt
        Lf += u_last * Vt * self.dxold
        
        while t1 - t.Get() > self.delta_t / 2:
            with TaskManager():
                if save:
                    gfut_out.AddMultiDimComponent(u_last.vec)
                Lf.Assemble()
                gfut.vec.data = self.PDE * Lf.vec
                RestrictGFInTime(spacetime_gf=gfut, reference_time=1.0, space_gf=u_last)
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
            gfut_out.AddMultiDimComponent(u_last.vec)
            return u_out, gfut_out
        else:
            return u_out
    
    def A(self, f, t0 = 0, t1 = 1):

        u = self.TimeStepping(source = f, t0 = t0, t1 = t1, save = False)
        return u

    def AT(self, u, t0 = 0, t1 = 1):

        F = self.TimeStepping(initial_cond = u, t0 = t0, t1 = t1, integrate = True, save = False)
        return self.fes_to_fes(F, fes2 = self.fes)
        
    def C(self, f):
        with TaskManager():
            v = self.fes.TestFunction()

            F = GridFunction(self.fes)
            F.vec.data = self.Prior * LinearForm(f * v * dx).Assemble().vec
        return F

    def CT(self, f):
        return self.C(f)
    
    def cov_to_trace(self, Cov):
        # Computes the trace of any arbitrary
        # ngsolve operator
        
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
    
    def cov_to_field(self, Cov, fes = None, squared = False):
        
        if fes is None:
            fes = self.fes0
        assert fes.globalorder == 0 and "l2".casefold() in fes.type.casefold(), "Must use 0th-order FES!"
        dia = np.empty(fes.ndof)
        self.create_Mass(fes)
        diagM = fes.MassMatrixDiagonal
        inds = np.flatnonzero(diagM)
        for i in inds:
            print("\r","Computing pointwise variance field index ",i,"/",len(inds),"...",sep="",end="")
            e = np.zeros(fes.ndof)
            e[i] = 1
            e = self.coeff_to_ngs(e, fes = fes)
            e = Cov(e)
            if squared:
                e = Cov(e)
            if e.space is not fes:
                self.create_Mass(fes,e.space)
                e = self.fes_to_fes(e, fes2 = fes)
            e = self.ngs_to_coeff(e)
            dia[i] = e[i]
        dia[inds] /= diagM[inds]
        dia = self.coeff_to_ngs(dia, fes = fes)
        return dia
    
    def sample(self, Ch):
        eta = self.rng.normal(size=self.n)
        eta = self.real_to_ngs(eta)
        sample = Ch(eta)
        return sample
