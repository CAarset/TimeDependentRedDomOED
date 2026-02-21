from runOED import *
from ngsolve.webgui import Draw
import ngsolve
from os import mkdir
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import dill as pickle
from vtk_to_png import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#global m
m = RDOED.m
mesh = mmaker.mesh
fes = mmaker.fes

R = RDOED.R
Q = RDOED.Q
A = mmaker.A
AT = mmaker.AT
C = mmaker.C
O = gmaker.O

mdigits = int(np.log10(m)+1)

graphics_path = "graphics"
design_folder = graphics_path + "/" + str(m) + "_shape_" + shape
try:
    mkdir(graphics_path)
except:
    pass
try:
    mkdir(design_folder)
except:
    pass

J_init = RDOED.J_init
J_init.eval(np.zeros(m))

fn = design_folder + "/" + "design"

#draw_stuff = True
draw_stuff = False
draw_cov = True

scale_fes = H1(mesh, order=1, definedon = "inner")
if draw_stuff:
    ### True source and data ###
    r = sqrt(x**2+y**2)
    phi = atan2(x,y)
    if 1:
        phi1 = (-4+0.0)*pi/5
        phi2 = (-2+0.0)*pi/5
        phi3 = (0+0.0)*pi/5
        phi4 = (2+0.0)*pi/5
        phi5 = (4+0.0)*pi/5

        offs = pi/8

        ang = lambda phil: IfPos(phi - (phil - offs),\
                IfPos((phil + offs) - phi,\
                    (phi - (phil - offs)) * ((phil + offs) - phi), 0), 0)

        cf = (mmaker.inner_radius - r)*(ang(phi1)+ang(phi2)+ang(phi3)+ang(phi4)+ang(phi5))
        cf = mmaker.C(cf)

        
        f_sca = GridFunction(scale_fes)
        fm_sca = GridFunction(scale_fes)
        
        if 0:
            ff = GridFunction(fes)
            ff.Set(cf)
            cf = AT(A(ff))
        f_sca.Set(cf)
        fm_sca.Set(-cf)
        
        f_max = np.max(f_sca.vec.FV().NumPy()[:])
        f_min = 0.034#-np.max(fm_sca.vec.FV().NumPy()[:])
    else:
        cf = exp(-100*r**2)
    #cf = CoefficientFunction(1)    
    f = GridFunction(fes)
    #f.Set((cf-f_min)/(f_max-f_min)-1/2)
    f.Set((cf - f_max)/f_max * -gmaker.peak * 45/19)
    #f.Set(cf)
    #f.Set((0.0349 + 0.011 - (cf + 0.011))/(0.0349 + 0.011)-1/2)

    #r = mmaker.ngs_to_real(f)
    #f = gmaker.CFT(gmaker.FC(r))
    #f = mmaker.real_to_ngs(f)
    #f = mmaker.AT(gmaker.OT(gmaker.O(mmaker.A(f))))
    #f = mmaker.AT(mmaker.A(f))
    u = A(f)
    g = O(u)

    r = mmaker.ngs_to_real(f)
    #r = Q@(R@(R.T@(Q.T@(r))))
    fr = mmaker.real_to_ngs(r)
    #f = fr
    gr = R.T@(Q.T@r)

#, floatsize = "single")
#vtk.Do()

#vtk_to_png(filename = fnn + ".vtk", \
#           imagename = fnn + ".png", \
#           scalar_range = (0,46),\
#           delete = True)

from os import listdir
from os.path import isfile, join
import gc

graphics_directory = design_folder
#

movie_length_s = 8
movie_resolution = 60#120
run_until = m# int((m)//2)
steplength = max(int(run_until // (movie_length_s * movie_resolution)),1)

step = 0
targets = []
objectives = []

if draw_stuff or draw_cov:
    nicefes = L2(mesh, order = 7, complex = False, dgjumps = True)
    gcf = GridFunction(nicefes)
    fn = graphics_directory + "/design"
        #num_str = str(target_number_of_sensors).zfill(mdigits)
        #fnn = fn + num_str
    vtk = VTKOutput(ma=mesh,coefs=[gcf],
        names=["designrecovery"],
        filename= fn,
        subdivision=5,
        legacy=True)
    vtk.Do()
    found_C_max = False

#def RedDom_to_png(target_number_of_sensors):
for target_number_of_sensors in range(1,run_until):#range(1, run_until, steplength):
    target_number_of_sensors = m
    
    RDOED.RedDom(target_number_of_sensors = target_number_of_sensors, verbose = False)
    out_flag = RDOED.out_flag
    
    wrd = RDOED.design
    
    if draw_stuff:
        
        mpst = RDOED.design_to_sol(w = wrd, f = f)
        scale = min(1,max(0.1,250*target_number_of_sensors/m))
        
        mpst = IfPos(mmaker.inner_radius**2 - x**2 - y**2, scale*((43.4391-1.55*mpst)*45/38.9-15)*45/66.3, 0)

    if draw_cov:
        Cpst = RDOED.design_to_cov(w = wrd)

        if not found_C_max:
            C_sca = GridFunction(scale_fes)
            C_sca.Set(Cpst)
            C_max = np.max(C_sca.vec.FV().NumPy()[:])
            found_C_max = True
        
    if draw_stuff or draw_cov:
        fn = design_folder + "/design"
        target_str = str(target_number_of_sensors).zfill(5)
        fnn = fn + target_str
        
        wcf = gmaker.grid_to_ngs(wrd)*2/3
        if draw_stuff:
            foo = mpst + wcf
        else:
            foo = gmaker.peak / C_max * Cpst + wcf
        gcf.Set(foo)
        vtk.Do()#time = target_number_of_sensors)

        step += 1
        vtk_to_png(filename = graphics_directory + "/design_step" + str(step).zfill(5) + ".vtk", \
                imagename = graphics_directory + "/" + target_str + ".png", \
                scalar_range = (0,gmaker.peak),\
                delete = True)#(0,46),\

        #del wrd, wcf, mpst #vtk

    else:
        objective = J_init.eval(wrd)
        targets.append(target_number_of_sensors)
        objectives.append(objective)
    gc.collect()
    #return target_number_of_sensors, objective
    


assert 0

if __name__ == '__main__':
    with Pool(processes=int(cpu_count()*3/4)) as pool:
        out = pool.map(RedDom_to_png, range(1, run_until, steplength))

targets = []
objectives = []
for outk in out:
    if outk[1] is not None:
        targets.append(outk[0])
        objectives.append(outk[1])
targets = np.array(targets)
objectives = np.array(objectives)
print(targets)
print(objectives)
sortd = np.argsort(targets)
print(sortd)
targets = targets[sortd]
objectives = objectives[sortd]

if not draw_stuff:
    plt.figure(figsize=(24,12))
    plt.plot(targets,objectives)
    plt.savefig("Objectives.png")
