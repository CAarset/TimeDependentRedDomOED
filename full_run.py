from runOEDmethod import *

t1 = 1
delta_t = 1e-1

rng = np.random.default_rng()
robin = 1
alpha = 0.001

target_m = int(4e2)
maxh = 0.03
refine_inner = 1

clear = False
compute_all = True

runOED(
           target_m = target_m, integration_order = 5, 
           maxh = maxh, order = 2, refine_inner = refine_inner,
           tol = 1e-15,
           t1 = t1, delta_t = delta_t,
           robin = robin, alpha = alpha,
           sensor_radius = None, verbose = False,
           clear = clear, compute_all = compute_all)
