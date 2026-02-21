#from juliacall import Main as jl
#from juliacall import Pkg as jlPkg
from functools import partial
from scipy.optimize import bisect
import numpy as np
from copy import deepcopy

#jlPkg.activate("pProject")
#jl.seval("using pProject")

import sys
sys.path.append('./Lp-ball-Projection')
from irbp_lib import get_lp_ball_projection as PProject

class PBoxProjections:
    def __init__(self, tol = 1e-15, proxtol = 1e-15, maxiter = int(1e5)):
        self.tol = tol
        self.proxtol = proxtol
        self.maxiter = maxiter

    def PNorm(self, x, p = 1):
        return np.sum(np.abs(x)**p)

    def PProject(self, x, radius = 1, p = 1):

        f = partial(self.PNorm, p = 1)
        
        # Instant return if in the p-ball
        if f(x) <= radius:
            return x
            
        if p == 1:
            prox = partial(self.PProx, x = x, p = 1)
            fprox = lambda lm: f(prox(lm = lm)) - radius

            a = 0
            b = np.amax(x)
            
            lm = bisect(fprox, a = a, b = b, xtol = self.tol, maxiter = self.maxiter)
            return prox(lm = lm)

        else:
            return PProject(starting_point = np.zeros_like(x), point_to_be_projected = x, p = p, radius = radius**p, \
                      tau=1.1, tol=self.tol, MAX_ITER=self.maxiter)
            #xProj = jl.convert(jl.Vector[jl.Float64],x)
            #print("\n",jl.pProject.norm(x,p))
            #xProj = jl.pProject.pProj(xProj, p, float(radius), tol = self.tol, proxtol = self.proxtol)
            #return np.array(xProj)
            
    def BoxProject(self, x):
        return np.fmin(np.fmax(0, x), 1)

    def PBoxProject(self, x, target, p = 1):
        xBox = self.BoxProject(x)
        f = partial(self.PNorm, p = 1)
        ptarget = float(target)**p

        if p == 1:
            if f(xBox) <= ptarget:
                return xBox
            
            prox = partial(self.PBoxProx, x = x, p = 1)
            fprox = lambda lm: f(prox(lm = lm)) - ptarget

            a = 0
            b = np.amax(x)
            
            lm = bisect(fprox, a = a, b = b, xtol = self.tol, maxiter = int(1e5))
            return prox(lm = lm)
        else:
            if np.all(np.logical_and(0 <= x,x <= 1)) and f(x) <= ptarget:
                return x
            return self.DouglasRachford(x = x, P1 = self.BoxProject, P2 = partial(self.PProject, radius = ptarget, p = p))

    def PBoxProx(self, x, p, lm):
        if p == 1:
            return self.BoxProject(np.array(self.PProx(x = x, p = 1, lm = lm)))
        else:
            raise ValueError("Not implemented!")

    def PProx(self, x, p, lm):
        if p == 1:
            return np.fmax(0, np.abs(x) - lm) * np.sign(x)
        else:
            return np.array([jl.pProject.pProx(jl.convert(jl.Float64,y), p, lm, tol = self.tol) for y in x])

    def DouglasRachford(self, x, P1, P2):
        xold = deepcopy(x)
        
        y = deepcopy(x)
        
        ydiff = np.inf
        diff = np.inf
        yzdiff = np.inf

        #R1 = lambda x: 2 * P1(x) - x
        #R2 = lambda x: 2 * P2(x) - x

        gamma_upper = np.sqrt(3/2) - 1

        gamma = 1e2 # gamma_upper * 2
        c0 = 1e3
        c1 = 1e5
        
        for i in range(self.maxiter):
            print("\rDR attempt ",i,"/",self.maxiter,", current diff: ",diff,", yzdiff: ",yzdiff,", gamma: ",gamma,sep="",end="")
            xold = deepcopy(x)

            yflag = False
            P1x = P1(x)
            while not yflag:
                yold = deepcopy(y)
                y = 1 / (1 + gamma) * (x + gamma * P1x)

                ydiff = np.linalg.norm(y - yold)

                yflag = ydiff < c0 / (i + 1) or np.linalg.norm(y) < c1

                gamma /= 1.1
                if not yflag:
                    y = deepcopy(yold)
            
            z = P2(2 * y - x)
            x = x + z - y
            
            #y = R1(x)
            #y = R2(y)
            
            #x = (x + y) / 2 #R2(R1(x))
            
            diff = np.linalg.norm(xold-x)
            if diff < self.tol:
                yzdiff = np.linalg.norm(y-z)
                if yzdiff < self.tol and gamma < gamma_upper: #i > 100 and 
                    #print(xold)
                    #print(x)
                    #if np.linalg.norm(x - P1(x)) < self.tol:
                        #if np.linalg.norm(x - P2(x)) < self.tol:
                    print("\nDR converged, breaking...")
                    break
        return P1(x)

if 0:
    def prox_psum_box(x,n,p):
        x = jl.convert(jl.Vector[jl.Float64],x)
        if p == 1:
            return prox_simplex_box(x,n)
        y = jl.pProject.pProj(x,p,float(n), tol = tol, proxtol = proxtol, maxiter = maxiter)
        return y
    
    M = 3000
    p = 1/3
    
    import numpy as np
    x = np.random.normal(size=M)
    x /= np.sum(np.abs(x)**p)**(1/p)
    x *= 30
    
    print(np.sum(np.abs(x)**p)**(1/p))
    
    y = prox_psum_box(x,2,p)
    #print(y)
    print(np.sum(np.abs(y)**p)**(1/p))
    print(jl.pProject.norm(y,p))
    
    y = prox_psum_box(y,2,p)
    #print(y)
    print(np.sum(np.abs(y)**p)**(1/p))
    print(jl.pProject.norm(y,p))
    
    print(float(2))