import numpy as np
import scipy
#import zfista

from copy import deepcopy
from time import time
from datetime import timedelta
from copy import deepcopy
import os

import random
from datetime import timedelta
import sys
import dill as pickle

from ngs_mesh import mesh_maker
from ngs_grid import grid_maker
from OptOED import OptOED
from RNGOED import RNGOED
from GreedyOED import GreedyOED

from ngsolve import IfPos, x, y

def runOED(target_m = int(4e2), integration_order = 5, 
           maxh = 0.025, order = 2, refine_inner = 1,
           tol = 1e-15,
           t1 = 10, delta_t = 1e-2,
           robin = 1, alpha = 1, scale = 1,
           sensor_radius = None, verbose = False,
           clear = True, \
           compute_all = True, \
           compute_optimal = False, compute_fields = False, compute_random = False, compute_greedy = False, **kwargs):

    compute_optimal = compute_all or compute_optimal
    compute_fields = compute_all or compute_fields
    compute_random = compute_all or compute_random
    compute_greedy = compute_all or compute_greedy

    random.seed(19)
    rng = np.random.default_rng(19)

    #T = np.linspace(t0, t1, time_steps)
    #dt = np.diff(T)[0]
    #print(dt)
    
    dim = 2
    scatterers = 3
    
    OED_types = ["Opt", "RedDom", "Greedy", "FISTA"]
    OED_type = OED_types[-1]
    
    noise_level = 1e-2 # This is relative noise level, in percent rel. to operator norm $FC$. Currently assuming identity noise cov (white noise).

            
    P = 2 # Stepping size for 0-norm approximation
    
    #mode = "qr"
    #mode = "auto"
    mode = "rsi"
    
    
    #shape = "gauss"
    #shape = "round"
    
    if sensor_radius is None:
        shape = "pointwise"
    else:
        shape = "gauss"
    
    PML = False
    
    if clear:
        answer = input("Delete old storage?")
        if answer.lower() in ["y","yes"]:
            for mydir in ["decomps","grid_dumps","ngs_dumps","opt_outputs"]:
                #try:
                for f in os.listdir(mydir):
                    os.remove(os.path.join(mydir, f))
                #except:
                #    pass
        elif answer.lower() in ["n","no"]:
            print("Skipping...")
        else:
            assert "Input should be y/n!"

    a = IfPos(y,5*y**2+y**3,0) + 1
    mmaker = mesh_maker(maxh = maxh, scatterers = scatterers, a = a, 
                         order = order, PML = PML, refine_inner = refine_inner, delta_t = delta_t, rng = rng, 
                         robin = robin, alpha = alpha, scale = scale)
    
    if mode == "qr":
        target_rank = None
    
    gmaker = grid_maker(mmaker, target_m = target_m, shape = shape, sensor_radius = sensor_radius, integration_order = integration_order)
    
    def target_rank(m,n):
        tr = max(min(m,150),\
                    3*max(\
                        int(10*float(m)**(1/4)),\
                        int(5*float(m)**(1/3))\
                    ))
        #target_rank = 233
        #target_rank = 100
        #tr = 200
        return min(min(tr, m), n)
    
    #target_rank = min(target_rank,63)#min(target_rank,40)
    start = time()
    
    kwargs = { \
              "mmaker": mmaker, "gmaker": gmaker,\
              "noise_level": noise_level, "tol": tol, "target_rank": target_rank, \
              "shape": shape, "mode": mode, "max_iters": 1e5, "P": P, \
              "verbose": verbose\
             }
    
    RDOED = OptOED(obs_times = [t1], **kwargs)
    RNG = RNGOED(obs_times = [t1], **kwargs)
    GREED = GreedyOED(obs_times = [t1], **kwargs)
    
    output_path = "opt_outputs"
    try:
        os.makedirs(output_path, exist_ok = True)
    except:
        pass
    fn = output_path + "/" + RDOED.solver + "_peps_" + str(RDOED.peps) + "_" + RDOED.target_filename
    RDOED.output_filename = fn
        
    stop = time()
    
    print("Red-Dom setup in " + str(timedelta(seconds=stop-start)) + "...")

    targets = np.concatenate((np.arange(1,37),np.array([72,108,RDOED.m_sensors])))
    if compute_optimal:
        # Compute optimal designs
    
        try:
            with open(fn, "rb") as filename:
                obj = pickle.load(filename)
            print("Successfully loaded previous results.")
        except:
            obj = {}
            
            obj["w1s"] = {}
            obj["ws"] = {}
            obj["wseqs"] = {}
            obj["ps"] = {}
            obj["times"] = {}
            
        w1 = None
        
        for target in targets:
            
            # Try to load optimal design from file
            if target in obj["ws"].keys():
                print("Skipping ",target,"...",sep="")
            else:
                start = time()
                RDOED.Opt(target = target, w1 = w1, verbose = False)
                time_taken = time() - start
                print("Finished for target ",target," in ",str(timedelta(seconds=time_taken)),"!",sep="")
                #w0 = deepcopy(RDOED.design)
                #w1 = deepcopy(RDOED.global_design)
                
                obj["w1s"][target] = deepcopy(w1)
                obj["ws"][target] = deepcopy(RDOED.design)
                obj["wseqs"][target] = deepcopy(RDOED.design_sequence)
                try:
                    obj["ps"][target] = deepcopy(RDOED.ps)
                except:
                    obj["ps"][target] = None
                obj["times"][target] = time_taken
            
                with open(fn, "wb") as filename:
                    pickle.dump(obj, filename)

    if compute_fields:
        # Compute pointwise variance fields of optimal designs

        fnFields = fn + "_fields"
        try:
            with open(fnFields, "rb") as filename:
                objFields = pickle.load(filename)
            print("Successfully loaded previous results.")
        except:
            objFields = {}
            
            objFields["fields"] = {}
            objFields["times"] = {}
            
        for target in targets:
            
            # Try to load field from file
            if target in objFields["fields"].keys():
                print("Skipping ",target,"...",sep="")
            else:
                start = time()
                field = RDOED.design_to_field(w = obj["ws"][target])
                time_taken = time() - start
                print("Finished field for target ",target," in ",str(timedelta(seconds=time_taken)),"!",sep="")
                
                objFields["fields"][target] = deepcopy(field)
                objFields["times"][target] = time_taken
            
                with open(fnFields, "wb") as filename:
                    pickle.dump(objFields, filename)
                    
    if compute_random:
        # Compute random designs
    
        fnRNG = fn + "_RNG"
        try:
            with open(fnRNG, "rb") as filename:
                objRNG = pickle.load(filename)
            print("Successfully loaded previous results.")
        except:
            objRNG = {}
            
            objRNG["ws"] = {}
            objRNG["allvals"] = {}
            objRNG["times"] = {}
            
        for target in targets:
            print("\r",target,end="")
            # Try to load optimal design from file
            if target in objRNG["ws"].keys():
                print("Skipping ",target,"...",sep="")
            else:
                start = time()
                RNG.RNG(target = target, tries = int(1e3), verbose = True)
                
                wRNG = deepcopy(RNG.design)
                allvals = deepcopy(RNG.allvals)
                
                objRNG["ws"][target] = deepcopy(wRNG)
                objRNG["allvals"][target] = deepcopy(allvals)
                objRNG["times"][target] = time() - start
            
                with open(fnRNG, "wb") as filename:
                    pickle.dump(objRNG, filename)

    if compute_greedy:
        GREED.Greedy(target_number_of_sensors = np.max(targets))
    
    return RDOED, RNG
