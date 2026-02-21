import numpy as np
from ngsolve import TaskManager, GridFunction

def pairg(myg):
    for i in myg:
        try:
            yield i + next(myg)
        except StopIteration:
            yield i

def comp(myg, length):
    for _ in range(int(np.ceil(np.log2(length)))):
        myg = pairg(myg)
    for i in myg:
        return i
    
def safe_add_CFs(CFs, weights = None, length = None):

    # Attempt to find the length of the CFs (not generally possible for generators, requiring length to be passed)
    if length is None:
        try:
            length = len(weights)
        except TypeError:
            try:
                length = len(CFs)
            except TypeError:
                raise Exception("If neither CFs nor weights have a length, please specify the length argument (generators do not generally have a defined length.")
    
    
    with TaskManager():
        if weights is not None:
            CFs = (cf*weight for cf, weight in zip(CFs, weights))
        else:  
            CFs = iter(CFs)
    return comp(CFs, length = length)

def interp_add_CFs(CFs, fes, weights = None):
    gf = GridFunction(fes)
    gf0 = GridFunction(fes)
    if weights is None:
        with TaskManager():
            for cf in CFs:
                gf0.Set(cf)
                gf.vec.data += gf0.vec.data
    else:
        with TaskManager():
            for cf, weight in zip(CFs, weights):
                if weight != 0:
                    gf0.Set(weight * cf)
                    gf.vec.data += gf0.vec.data
    return gf