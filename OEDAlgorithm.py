import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, LinearOperator, cg
from scipy.linalg import solve
from scipy.linalg import cholesky as chol

from lrQR import *
from LRtrace import trace
import dill as pickle
from copy import deepcopy
#from ngsolve import TaskManager
from util import power_iteration_matrix
from create_decomp import *

from ngsolve import Integrate, CoefficientFunction, GridFunction

from os import listdir, makedirs
from os.path import isfile, join
from time import time
from datetime import timedelta

class OEDAlgorithm:
    def __init__(self, mmaker, gmaker,\
                    obs_times = [1], \
                    noise_level = 5e-3, tol = 1e-15, max_iters = int(1e3), \
                    jtol = 1e-2, \
                    shape = "round", target_rank = 100, mode = "automatic", \
                    verbose = True, samples = int(1e3), noise_type = "individual", **kwargs):
        
        self.rng = np.random.default_rng(199)
        
        self.w = None
        self.w_old = None
        
        # Grab mesh- and grid makers
        self.mmaker = mmaker
        self.gmaker = gmaker

        # Work out actual problem dimensions
        
        self.n = mmaker.n
        
        self.obs_times = np.sort(np.unique(obs_times))
        self.m_obs = len(obs_times)
        
        self.m_sensors = self.gmaker.m_sensors
        self.m = int(self.m_sensors * self.m_obs)
        
        # Grab operators

        self.A = partial(mmaker.A, t1 = obs_times[-1])
        self.AT = partial(mmaker.AT, t1 = obs_times[-1])
        self.F = partial(gmaker.F, obs_times = self.obs_times)
        self.FT = partial(gmaker.FT, obs_times = self.obs_times)

        self.delta_t = self.mmaker.delta_t
        
        self.tol = tol # Tolerance for solvers
        self.jtol = jtol # Relative tolerance for numerical ordering of gradient
        self.max_iters = max_iters
        self.target_rank = target_rank(m = self.m, n = self.n)
        self.mode = mode
        self.shape = shape
        self.samples = samples
        self.noise_type = noise_type
        self.noise_level = noise_level
        
        self.current_m = deepcopy(self.m)
        self.current_free = deepcopy(self.m_sensors)
        self.current_target = deepcopy(self.m_sensors)
        
        self.design = np.zeros(self.m_sensors)

        self.free_indices = np.arange(self.m_sensors, dtype = int)
        self.dom_indices = np.array([], dtype = int)
        self.red_indices = np.array([], dtype = int)

        self.fixed_doms = 0
        self.fixed_reds = 0

        self.out_flag = [0,0,"Failed"]
        self.verbose = verbose
        
        self.actual_spectrum = None
        self.full_spectrum = None
        
        def vprint(str):
            if self.verbose:
                print(str)
        self.vprint = vprint

        outputs_directory = "opt_outputs/"
        makedirs(outputs_directory, exist_ok = True)
        self.outputs_directory = outputs_directory
        
        decomps_dumps_directory = "decomps/"
        makedirs(decomps_dumps_directory, exist_ok = True)
        self.decomps_dumps_directory = decomps_dumps_directory

        target_filename = "msensor_" + str(self.m_sensors) + "_mobs_" + str(self.m_obs) + \
                          "_obs_times_" + str(self.obs_times) + \
                          "_noisetype_" + str(self.noise_type) + "_noiselevel_" + str(round(-np.log10(self.noise_level),2)) + \
                          "_" + self.gmaker.target_filename
        self.target_filename = target_filename
        
        load_flag = False
        # Attempt to find a stored decomposition with the same target
        for filename in listdir(decomps_dumps_directory):
            if isfile(join(decomps_dumps_directory, filename)):
                if target_filename in filename:
                    with open(decomps_dumps_directory + filename, "rb") as input_file:
                        obj = pickle.load(input_file)
                        self.vprint("Successfully loaded stored decomposition!")
                        self.ell = obj["ell"]
                        self.R = obj["R"]
                        self.Q = obj["Q"]
                        
                        self.full_spectrum = obj["full_spectrum"]
                        
                        self.norm = obj["norm"]
                        
                        self.basis_change_matrix = obj["basis_change_matrix"]
                        self.CQmat = obj["CQmat"]
                        
                        self.offset = obj["offset"]
                        self.decomptime = obj["decomptime"]
                        self.data_variance = obj["data_variance"]
                        self.full_data_variance = obj["full_data_variance"]
                        load_flag = True
                        break
        if not load_flag:
            filename = decomps_dumps_directory + target_filename
            self.vprint("Data file " + filename + " does not exist, creating decomposition " + \
                         "with target " + str(self.target_rank) + "...")
            
            # Estimation of data size for relative noise level purposes.
            self.full_data_variance = np.zeros(self.m)
            
            samples = self.samples

            for i in range(int(samples)):
                print("\rEstimating data variance, step ",i,"/",samples,"...",sep="",end="")
                s = self.mmaker.sample(mmaker.C)
                #r = self.mmaker.ngs_to_real(s)
                #us = self.mmaker.A(s)
                gs = self.gmaker.F(s, obs_times = self.obs_times) #self.gmaker.O(us)
                self.full_data_variance += np.abs(gs)**2 / samples
            print("")
            
            if self.noise_type.casefold() == "exp".casefold():
                max_data_variance = np.max(self.full_data_variance)
                min_data_variance = np.min(self.full_data_variance)

                x_norms = np.linalg.norm(self.grid, axis = 1)

                a = (np.log(max_data_variance)-np.log(min_data_variance)) / (np.max(x_norms) - np.min(x_norms))
                b = max_data_variance * np.exp(a * np.min(x_norms))

                self.data_variance = b * np.exp(-a * x_norms)
            elif self.noise_type.casefold() == "quad".casefold():
                max_variant_sensor = np.argmax(self.full_data_variance)
                max_data_variance = self.full_data_variance[max_variant_sensor]

                x_norms = np.linalg.norm(self.grid, axis = 1)

                a = max_data_variance * x_norms[max_variant_sensor]**2
                
                self.data_variance = a / x_norms**2
            elif self.noise_type.casefold() == "average".casefold():
                self.data_variance = np.mean(self.full_data_variance)
            else:
                self.data_variance = self.full_data_variance #(self.full_data_variance + np.mean(self.full_data_variance)) / 2

            if 0:
                self.ell = 10
                self.Q = self.rng.normal(size = (self.n,self.ell))
                self.R = self.rng.normal(size = (self.ell,self.m))
                self.actual_spectrum = None
                self.full_spectrum = None
                self.decomptime = 0
            
            else:
                # Create QR decomposition
                decomp_maker = create_decomp(F = self.Forward, FT = self.Adjoint, \
                                             m = self.m, n = self.n, \
                                             tol = self.tol, target_rank = self.target_rank, \
                                             mode = self.mode)
                decomp_maker.decomp()
                
                self.Q = decomp_maker.Q
                self.R = decomp_maker.R
                self.ell = decomp_maker.ell
                self.actual_spectrum = decomp_maker.actual_spectrum
                self.full_spectrum = decomp_maker.full_spectrum
                self.decomptime = decomp_maker.decomptime
            
            Q, R = self.Q, self.R
            
            self.vprint("Dimension reduced from " + str((self.m, self.n)) + \
                        " to " + str(self.ell) + " in " + str(timedelta(seconds = self.decomptime)))

            # Norm estimation to make use of relative noise level
            norm_square = power_iteration_matrix(lambda x: Q@(R@(R.T.conj()@(Q.T.conj()@x))), n = self.n, iters = 100)
            self.norm = np.sqrt(norm_square)
            print("Forward operator norm estimate:",self.norm, "vs.", self.data_variance)
            
            # The basis change matrix QC_{prior}^2Q^T
            
            with TaskManager():
                Cmat = np.empty((self.n,self.ell))
                CQmat = np.empty((self.n,self.ell))
                for i in range(self.ell):
                    CQ = self.mmaker.real_to_ngs(Q[:,i])
                    CQ = self.mmaker.C(CQ)
                    CQmat[:,i] = self.mmaker.ngs_to_real(CQ)
                    CQ = self.mmaker.C(CQ)
                    Cmat[:,i] = self.mmaker.ngs_to_real(CQ)
                Cmat = Q.T@Cmat
            
            self.CQmat = CQmat    
            self.basis_change_matrix = Cmat
            
            self.offset = mmaker.tracePrior - np.trace(self.basis_change_matrix)
            
            if self.target_filename is not None:
                obj = {"ell": self.ell, \
                       "R": self.R, "Q": self.Q, \
                       "norm": self.norm, "full_spectrum": self.full_spectrum, \
                       "basis_change_matrix": self.basis_change_matrix, "CQmat": self.CQmat, \
                       "offset": self.offset, "decomptime": self.decomptime, \
                       "data_variance": self.data_variance, "full_data_variance": self.full_data_variance}
                with open(filename, "wb") as output_file:
                    pickle.dump(obj, output_file)    
                self.vprint("Successfully stored decomposition.")

            
        # Scale R based on desired relative noise level (does not require re-run).
        
        # Scale R based on desired relative noise level (does not require re-run).

        self.R_unscaled = self.NoiseCovHalf(self.R)
        self.R = 1 / np.sqrt(self.noise_level) * self.R        
        
        J = trace(Q = self.Q, R = self.R, W = self.W, m_sensors = self.m_sensors, m_obs = self.m_obs, \
                  basis_change_matrix = self.basis_change_matrix, Roffset = None, offset = self.offset)
        self.J_init = J
        self.J = J
        
        self.Bh = self.J_init.basis_change_matrix_half
        self.Bh0 = self.J_init.basis_change_matrix_half
    
    def W(self, w):
        return np.tile(w,self.m_obs)
    
    def NoiseCovHalf(self, g, include_noise_level = False):
        g_noise = g * np.sqrt(self.data_variance)
        if include_noise_level:
            g_noise *= np.sqrt(self.noise_level)
        return g_noise
        
    def NoiseCovInverseHalf(self, g, include_noise_level = False):
        #g_noise = g / np.sqrt(self.data_variance)
        # Safety net against zero sensor measurements leading to zero variance
        g_noise = np.divide(g, np.sqrt(self.data_variance), out=np.zeros_like(g), where=self.data_variance!=0)
        if include_noise_level:
            g_noise /= np.sqrt(self.noise_level)
        return g_noise
    
    def Forward(self, X, cycle = 0):
        Y = np.empty((self.m, X.shape[1]))
        #with TaskManager():
        for i in range(X.shape[1]):
            print("\rForward applying ",i,"/",X.shape[1],"...",sep="",end="")
            x = X[:,i]
            Y[:,i] = self.NoiseCovInverseHalf(self.gmaker.FC(x, obs_times = self.obs_times))

            # Save output, as this will take some time.
            #obj = {"X": X, "cycle": cycle, "i": i, "Y": Y}
            #with open("forwardtemp", "wb") as output_file:
            #    pickle.dump(obj, output_file)    
        print("")
        return Y

    def Adjoint(self, Y, cycle = 0):
        X = np.empty((self.n, Y.shape[1]))
        #with TaskManager():
        for i in range(Y.shape[1]):
            print("\rAdjoint applying ",i,"/",Y.shape[1],"...",sep="",end="")
            g = Y[:,i]
            X[:,i] = self.gmaker.CFT(self.NoiseCovInverseHalf(g), obs_times = self.obs_times)
            
            # Save output, as this will take some time.
            #obj = {"X": X, "cycle": cycle, "i": i, "Y": Y}
            #with open("adjointtemp", "wb") as output_file:
            #    pickle.dump(obj, output_file)    
        print("")
        return X
    
    def design_to_cov(self, w, fac = 1, half = False, skip_prior = False):
        
        Q = self.Q
        CQ = self.CQmat
        R = self.R
        mmaker = self.mmaker
        fes = mmaker.fes
        C = mmaker.C

        LIB = (R*self.W(w)) @ R.T.conj() + fac * np.eye(self.ell)
        
        if skip_prior:
            LIB = solve(LIB, CQ.T.conj(), check_finite=True, assume_a='pos')
            LIB = LIB - CQ.T.conj()
        
            def cov(f):
            
                # Catch bad input (must be GridFunction in our standard source fes)
                check = isinstance(f, CoefficientFunction) and not isinstance(f, GridFunction)
                if not check and isinstance(f, GridFunction):
                    if f.space != fes:
                        check = True
                        
                if check:
                    F = GridFunction(fes)
                    F.Set(f)
                    F = mmaker.ngs_to_real(F)
                else:
                    F = mmaker.ngs_to_real(f)
                    
                F = LIB @ F
                F = CQ @ F
                F = mmaker.real_to_ngs(F)
                return F
                
        else:
            LIB = solve(LIB, Q.T.conj(), check_finite=True, assume_a='pos')
            LIB = LIB - Q.T.conj()
        
            def cov(f):
                F0 = C(f)
                F = mmaker.ngs_to_real(F0)
                F = LIB @ F
                F = Q @ F
                F = mmaker.real_to_ngs(F)
                F = C(F+F0)
                return F

        if half:
            cov_decomp = create_decomp(F = cov, FT = cov, m = self.n, n = self.n, \
                         tol = self.tol, target_rank = self.target_rank, \
                         mode = self.mode)
            cov_decomp.do()
            
            return lambda x: cov_decomp.SVD["U"]@(np.sqrt(cov_decomp.SVD["S"]) * (cov_decomp.SVD["Vh"]@x))
        
        return cov
        
    def design_to_Hess(self, w, fac = 1):
        
        mmaker = self.mmaker
        gmaker = self.gmaker
        W = self.W
        
        # Exact Hessian
        C = mmaker.C
        F = self.F
        FT = self.FT
        Sghi = partial(self.NoiseCovInverseHalf, include_noise_level = True)
        
        def Hess(x):
            f = mmaker.real_to_ngs(x)
            f = C(f)
            g = F(f)
            g = Sghi(g)
            g = fac * W(w) * g
            g = Sghi(g)
            f = FT(g)
            f = C(f)
            return mmaker.ngs_to_real(f) + x

        # Low-rank Hessian inverse for preconditioner
        Q = self.Q
        R = self.R
            
        LIB = (R*W(w)) @ R.T.conj() + fac * np.eye(self.ell)
          
        def HessLRInv(x):
            y = Q.T @ x
            y = solve(LIB, y, check_finite=True, assume_a='pos') - y
            y = Q @ y
            return x + y

        return LinearOperator(matvec = Hess, shape = (self.n, self.n)), LinearOperator(matvec = HessLRInv, shape = (self.n, self.n))
        
    def design_to_cov_exact(self, w, fac = 1):
        
        mmaker = self.mmaker
        C = mmaker.C
        
        Hess, HessLRInv = self.design_to_Hess(w = w, fac = fac)   

        def cov(f):
            fc = C(f)
            fc = mmaker.ngs_to_real(fc)
            fc, _ = cg(A = Hess, b = fc, x0 = HessLRInv(fc), rtol = 1e-25, M = HessLRInv)
            fc = mmaker.real_to_ngs(fc)
            fc = C(fc)
            return fc
        return cov
    
    def design_to_field(self, w, interp = False):
        
        # Build current cov
        self.current_cov = self.design_to_cov(w, skip_prior = True)
        
        # Build cov field (sans prior)
        cfield = self.mmaker.diag(self.current_cov, order = self.ell, interp = interp)
        
        # Correct by reintroducing the prior field
        if isinstance(cfield, GridFunction) and isinstance(self.mmaker.diagPrior, GridFunction):
            cfield.vec[:] += self.mmaker.diagPrior.vec[:]
        else:
            if interp:
                cfieldGF = GridFunction(cfield.space)
                cfieldGF.Set(cfield + self.mmaker.diagPrior)
                return cfieldGF
            cfield += self.mmaker.diagPrior
        return cfield
        
    def design_to_trace(self, w, exact = False):
        
        # Build current cov
        if exact:
            self.current_cov = self.design_to_cov_exact(w = w)
        else:
            self.current_cov = self.design_to_cov(w = w, skip_prior = True)
        
        # Build trace (if LR-approximate, sans prior, which we add later)
        tr = self.mmaker.cov_to_trace(self.current_cov)
        
        if not exact:
            # Correct by reintroducing the prior trace (LR only)
            tr += self.mmaker.tracePrior
        return tr
        
    def design_to_sol(self, w, r = None, f = None, u = None, g = None, add_noise = False, fac = 1):
        assert (r is not None) + (f is not None) + (u is not None) + (g is not None) == 1, "Exactly one of r, f, u and g should be specified!"
        
        mmaker = self.mmaker
        gmaker = self.gmaker
        
        if r is not None:
            print("Computing f from r...")
            f = mmaker.real_to_ngs(r)
        if f is not None:
            print("Computing g from f...")
            g = gmaker.F(f, obs_times = self.obs_times)
        elif u is not None:
            print("Computing g from u...")
            g = gmaker.O(u)
        
        if add_noise:
            raise Exception("Noise addition not currently supported.")
        
        self.current_cov = self.design_to_cov(w, skip_prior = False, fac = fac)    
        Sghi = partial(self.NoiseCovInverseHalf, include_noise_level = True)
        
        reco = Sghi(self.W(w) * g)
        reco = Sghi(reco)
        print("Backpropagating data...")
        reco = gmaker.FT(reco, obs_times = self.obs_times)
        reco = self.current_cov(reco)
        
        return reco
        
    def design_to_sol_exact(self, w, r = None, f = None, u = None, g = None, add_noise = False, fac = 1):
        assert (r is not None) + (f is not None) + (u is not None) + (g is not None) == 1, "Exactly one of r, f, u and g should be specified!"
        
        gmaker = self.gmaker
        self.current_cov = self.design_to_cov_exact(w = w, fac = fac)

        F = partial(gmaker.F, obs_times = self.obs_times)
        FT = partial(gmaker.FT, obs_times = self.obs_times)
        Sghi = partial(self.NoiseCovInverseHalf, include_noise_level = True)
        
        if r is not None:
            print("Computing f from r...")
            f = mmaker.real_to_ngs(r)
        if f is not None:
            print("Computing g from f...")
            g = gmaker.F(f, obs_times = self.obs_times)
        elif u is not None:
            print("Computing g from u...")
            g = gmaker.O(u)
        
        if add_noise:
            raise Exception("Noise addition not currently supported.")
        
        reco = Sghi(g)
        reco = self.W(w) * reco
        reco = Sghi(reco)
        reco = FT(reco)
        
        reco = self.current_cov(reco)
        return reco
        
    def Jac(self, w, full = False):
        
        if full:
            LIB = (self.R*self.W(w))@self.R.T.conj() + np.eye(self.ell)
            LIB = solve(LIB, self.Bh0, check_finite=True, assume_a='pos')
            
            norms = np.linalg.norm(LIB.T.conj()@self.R, axis=0)**2
            
            return -np.array([np.sum(norms[i::self.m_sensors]) for i in range(self.m_sensors)])
            
        else:
            LIB = (self.Rfree*self.W(w[self.free_indices]))@self.Rfree.T.conj() + self.I_plus_RdomR
            LIB = solve(LIB, self.Bh, check_finite=True, assume_a='pos')
            
            norms = np.linalg.norm(LIB.T.conj()@self.Rfree, axis=0)**2
            raise Exception("Not properly implemented.")
            return
        
    def update(self, dominant_indices = np.array([], dtype = int), redundant_indices = np.array([], dtype = int)):
        assert not np.intersect1d(dominant_indices, redundant_indices).size, "Indices cannot be both dominant and redundant!"
        self.dom_indices = np.union1d(self.dom_indices, dominant_indices)
        self.red_indices = np.union1d(self.red_indices, redundant_indices)
        self.free_indices = np.setdiff1d(self.free_indices, np.union1d(self.dom_indices,self.red_indices))
        self.current_free = self.free_indices.size
        
        wfree = np.zeros(self.m_sensors)
        wfree[self.free_indices] = 1
        self.Rfree = self.R[:,np.argwhere(self.W(wfree)).ravel()]

        wdom = np.zeros(self.m_sensors)
        wdom[self.dom_indices] = 1
        self.Rdom = self.R[:,np.argwhere(self.W(wdom)).ravel()]

        self.I_plus_RdomR = np.eye(self.ell)
        if self.dom_indices.size:
            self.I_plus_RdomR += self.Rdom@self.Rdom.T

        self.J = trace(Q = self.Q, R = self.Rfree, W = self.W, m_sensors = self.current_free, m_obs = self.m_obs, \
                       basis_change_matrix = self.basis_change_matrix, Roffset = self.I_plus_RdomR, offset = self.offset)
        self.B = self.J.basis_change_matrix
        self.Bh = self.J.basis_change_matrix_half

        #self.current_target = self.target_number_of_sensors - len(self.dom_indices)
        self.current_m = self.m_sensors - len(self.dom_indices) - len(self.red_indices)
        self.fixed_doms += len(dominant_indices)
        self.fixed_reds += len(redundant_indices)
        if 0:#self.current_free:
            C = np.linalg.norm(self.B, ord = 2) * np.linalg.norm(self.Rfree@self.Rfree.T, ord = 2)**2
            self.C0 = np.sqrt(self.current_target) * C
            self.C1 = np.sqrt(self.current_m - self.current_target) * C
            self.C2 = np.sqrt(2 * self.current_target) * C
            self.vprint("C0: " + str(self.C0) + ", C1: " + str(self.C1) + ", C2: " + str(self.C2))
        return
    
    def test_optimality(self, w, J = None, full = False, current_target = None, exit = True):

        if current_target is None:
            if full:
                current_target = int(np.sum(w!=0))
            else:
                current_target = self.current_target
            
        if current_target == 0:
            if np.all(w==0):
                return True
            else:
                return False
            
        if J is None:
            J = self.Jac(w, full = full)
           
        #largest_grad_components = np.argpartition(-J, -current_target)[-current_target:]
        nth_largest_grad_component = np.argpartition(-J, -current_target)[-current_target]
        large_grad_components = np.argwhere(-J > (nth_largest_grad_component + np.sqrt(self.tol)))
        
        if full:
            wfree = w
        else:
            wfree = w[self.free_indices]
        active_sensors = np.nonzero(wfree)
        
        ## If the sets are the same, optimality applies.
        #if not np.setdiff1d(largest_grad_components,active_sensors).size and not np.setdiff1d(active_sensors,largest_grad_components).size:
        
        # Actually, it is enough that the active sensors correspond to any large grad components.
        if not np.setdiff1d(active_sensors,large_grad_components).size:
            self.vprint("Active sensors: " + str(active_sensors))
            self.vprint("Large grad components: " + str(large_grad_components))
            if exit:
                self.vprint("Global minimum found by optimality criterion, exiting!")
                self.out_flag[2] = "Optimality"
                self.update(dominant_indices = np.nonzero(w), redundant_indices = np.nonzero(w==0))
                self.fixed_doms = len(self.dom_indices)
                self.fixed_reds = len(self.red_indices)
            return True
        return False
    
    def nonb(self, w):
        return np.mean(w*(1-w))

    def set_binary(self, w, target, return_jac = False, sort = False, set_red_dom = False):
        # If w is assumed to be the globally optimal solution of the 1-relaxed sensor placement problem,
        # then it will typically have a large number of exactly-0 (and possibly some exactly-1) weights
        # corresponding to small resp. large gradient components.
        # This algorithm identifies such indices, and corrects any that have been set numerically close
        # to, but not exactly equal to, 0 or 1.

        j = self.J_init.jac(w)
        order = np.argsort(j)
        jo = j[order]
        if target == 0:
            jom0 = -np.inf
        else:
            jom0 = jo[target-1]

        if sort:
            w = w[order]
            j = jo
            
        dom = (w > 1 - self.tol) & ((j - jom0)/np.abs(jom0) <  - self.jtol)
        red = (w < self.tol) & ((j - jom0)/np.abs(jom0) > self.jtol)
        free = np.logical_not(dom) & np.logical_not(red)

        dom = np.argwhere(dom)
        red = np.argwhere(red)
        free = np.argwhere(free)

        w[dom] = 1
        w[red] = 0

        if set_red_dom:
            self.dom_indices = dom
            self.red_indices = red
            self.free_indices = free
            self.vprint("Setting " + str(dom.size) + " dominant and " + str(red.size) + " redundant indices from global solution...")    
            self.update()

        if return_jac:
            return w, j
        else:
            return w

    def insert(self, w):
        wout = np.zeros(self.m_sensors)
        wout[self.free_indices] = w
        wout[self.dom_indices] = 1
        return wout
        
    def Fd(self, f, include_noise = False):
        
        if include_noise:
            R = self.R
        else:
            R = self.R_unscaled
            
        r = self.mmaker.ngs_to_real(f)
        g = R.T@(self.Q.T@r)
        return g

    def FdT(self, g, include_noise = False):
        if include_noise:
            R = self.R
        else:
            R = self.R_unscaled

        r = self.Q@(R@g)
        f = self.mmaker.real_to_ngs(r)
        return f
