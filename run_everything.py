import numpy as np
import scipy
import zfista

from copy import deepcopy
from time import time
from datetime import timedelta
from copy import deepcopy

from ngs_grid import *
from ngs_mesh import *
from RedDomOED import *
from GreedyOED import *
from FISTAOED import *

from itertools import product

import random
random.seed(19)
rng = np.random.default_rng(19)

dim = 2
scatterers = 3

orders = [0,1,2]
ms = [10,50,100,200,300,500,1000,2000,5000]
noise_level = 1e-2 # This is relative noise level, in percent rel. to operator norm $FC$. Currently assuming identity noise cov (white noise).
tol = 1e-5
maxh = 0.03
omegass = [[20], [25], [30], [35], [40], [45], [50], [20, 30, 40, 50], [40, 50], [20, 25, 30, 35, 40, 45, 50]]
mode = "rsi"

shapes = ["pointwise","gauss","round"]

PML = False
refine_inners = [0,1]

for do in [0,1]:
    for order, m, shape, omegas, refine_inner in product(orders, ms, shapes, omegass, refine_inners):
        mmaker = mesh_maker(maxh = maxh, scatterers = scatterers, \
                             order = order, omegas = omegas, PML = PML, refine_inner = refine_inner)
        mesh = mmaker.mesh

        gmaker = grid_maker(mmaker, target_m = m, shape = shape)

        n = mmaker.n
        m = gmaker.m

        if do:
            
            target_ranks = np.concatenate((np.array([5]),np.arange(10,min(n,m)+1,10)))

            for target_rank in target_ranks:
                kwargs = { \
                          "mmaker": mmaker, "gmaker": gmaker,\
                          "noise_level": noise_level, "tol": tol, "target_rank": target_rank, \
                          "shape": shape, "mode": mode, "max_iters": 1e5 \
                         }

                RDOED = RedDomOED(**kwargs)
