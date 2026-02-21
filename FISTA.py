from OEDAlgorithm import *
from scipy.optimize import bisect
from time import time, sleep
from IPython.display import display, clear_output

def prox_l1(x,alpha):
    return np.fmax(np.abs(x)-alpha,0)*np.sign(x)

def prox_unit_cube(x,alpha):
    return np.fmax(np.fmin(x,1),0)

def prox_l1_unit_cube(x,alpha):
    prox = prox_l1(x,alpha)
    prox = prox_unit_cube(prox,None)
    return prox

def prox_simplex_box(x,n): # Projects onto [0,1]^d intersected with the (filled) simplex scaled by a factor n (i.e. 0<=x<=1 and sum(x)<=n).
    assert isinstance(n,int) and n<=len(x), "n must be an integer no greater than the length of the input x \
        (currently n = " + str(n) + ", len(x) = " + str(len(x)) + ")..."
    y = prox_unit_cube(x,None)
    if np.sum(y)<=n:
        return y
    else:
        f = lambda lam: np.sum(prox_unit_cube(x - lam,None)) - n
        lam = bisect(f,0,np.amax(x))
        return prox_unit_cube(x - lam,None)

def prox_l1_simplex_box(x,alpha,n):
    prox = prox_l1(x,alpha)
    prox = prox_simplex_box(prox,n)
    return prox

class FISTA:#(OEDAlgorithm):
    def __init__(self, f, jac, prox, max_iters = 1e5, tol = 1e-15, eta = 1.1, convex = True, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        
        self.max_iters = int(max_iters)
        self.max_iters_digits = int(np.ceil(np.log10(self.max_iters))+1)
        self.tol = tol
        
        self.x_old_old = None
        self.x_old = None
        self.x = None
        
        self.x_diff_old = np.inf
        self.x_diff = np.inf
        
        self.eta = eta
        
        self.f = f
        self.jac = jac
        
        self.fx_old = np.inf
        self.fx = np.inf
        self.stored_grad = None

        self.prox = prox
        self.convex = convex
        self.instant_out = False
        
        self.penalty = 1
        
    def step(self, output = False):

        try:
            xprox = self.prox(tau = self.IL, x = self.x - self.IL * self.grad())
        except:
            print(self.grad())
            xprox = self.x # Bad hack
        #print(self.grad())
        #print(self.x)
        if output:
            return xprox
        self.x_old_old = deepcopy(self.x_old)
        self.x_old = deepcopy(self.x)
        self.x = deepcopy(xprox)
        
        if self.convex:
            self.x_diff_old = deepcopy(self.x_diff)
            self.x_diff = np.linalg.norm(self.x_old-self.x)
            if self.x_diff < self.tol:
                print("No change (" + str(self.x_diff) + "), terminating...")
                print(self.x[self.x!=0])
                print(self.grad()[self.x!=0])
                return True
        return False
        
    def update_grad(self):
        do = False
        if self.x_old is None:
            do = True
        elif not np.all(self.x == self.x_old):
            do = True
        if do:
            self.stored_grad = self.jac(self.x)
            
    def grad(self, x = None):
        if x is None:
            self.update_grad()
            return self.stored_grad
        else:
            return self.jac(x)
    
    #def look_ahead(self):
    #    J = self.grad()
    #    wJ = np.zeros(self.m_sensors)
    #    grad_components = np.argpartition(-J, -self.target_number_of_sensors)[-self.target_number_of_sensors:]
    #    wJ[grad_components] = 1
    #    Jw = self.grad(wJ)
    #    optimality = self.test_optimality(w = wJ, J = Jw)
    #    if optimality:
    #        self.x = wJ
    #    return optimality, wJ, Jw
    
    def backtrack(self):
        #self.L *= eta
        #return 
        self.fx_old = deepcopy(self.fx)
        self.fx = self.f(self.x)
        while True:
            
            px = self.step(output = True)
        
            Q1 = self.fx + np.inner(px - self.x,self.grad())
            Q2 = np.linalg.norm(px - self.x)**2
            F = self.f(px)
        
            Q = Q1 + 1/(2*self.IL) * Q2
            if F <= Q:
                break
            self.IL /= self.eta
            #self.check_frequency = max(1, int(self.check_frequency//2))
            
    def solve(self, x0, L, backtrack = False, verbose = False):
        
        self.verbose = verbose
        self.x = deepcopy(x0)
        
        res = {}
        res["success"] = False
        
        if not callable(L) and not type(L) is str:
            self.IL = 1 / L

        for i in range(self.max_iters):
            
            if L == "jac_ordering":
                jac = self.grad()
                inds = np.logical_and(self.x < 1, jac != 0)
                try:
                    self.IL = 2 / np.nanmin(-(1-self.x[inds])/jac[inds])
                    self.IL /= self.penalty
                except:
                    res["x"] = self.x
                    res["success"] = True
                    return res
                
            if callable(L):
                self.IL = 1 / L(i)

            stop = self.step()
            if stop:
                if not i:
                    self.instant_out = True
                res["x"] = self.x
                res["success"] = True
                return res

            if backtrack:
                if callable(L) and backtrack > 1 and i >= backtrack:
                    L = None
                    self.IL = 1
                if not callable(L):
                    self.backtrack()
                    
            if self.x_diff_old < self.x_diff or self.fx_old < self.fx:
                #self.IL /= self.eta
                self.penalty *= self.eta
                #if self.convex:
                self.x = deepcopy(self.x_old)
                self.x_old = deepcopy(self.x_old_old)

            #self.L = min(self.L*(1.1 + np.all(self.x==self.x_old)),self.Lmin)
            #if not i%self.check_frequency:
            #    self.check_frequency = max(1,self.check_frequency-1)
            #    optimality, wJ, Jw = self.look_ahead()
            #    if optimality:
            #        print("Optimality criterion reached for i = ",i,", terminating...",sep="")
            #        break
            #    nonz = np.nonzero(wJ)
            #    grad_components = np.argpartition(-Jw, -self.target_number_of_sensors)[-self.target_number_of_sensors:]
            #    clear_output(wait=True)
            
            if not backtrack:
                self.fx_old = deepcopy(self.fx)
                self.fx = self.f(self.x)
            if not self.convex:
                self.x_diff_old = deepcopy(self.x_diff)
                self.x_diff = np.linalg.norm(self.x_old-self.x)
                
            print("\r" + str(i).zfill(self.max_iters_digits) + "/" + str(self.max_iters) + \
                    ", L = " + str(1 / self.IL) + ", active w = " + str(np.sum(self.x!=0)) + \
                    ", fval = " + str(self.fx) + \
                    ", norm change: " + str(self.x_diff) + \
                    ", penalty: " + str(self.penalty) + "...", sep="",end="")#, \
                  #"Components: " + str(np.sort(grad_components)), \
                  #"Active: " + str(nonz))
            #sleep(0.001)
                
        res["x"] = self.x
        return res