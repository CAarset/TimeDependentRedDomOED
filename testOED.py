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

from os import listdir
from os.path import isfile, join

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

tests_folder = "tests/"
try:
    mkdir(tests_folder)
except:
    pass

RDOED.silent = True
J_init = RDOED.J_init
J_init.eval(np.zeros(m))

prefac = 1e-5

def testRedDom(target_number_of_sensors):
    
    RDOED.RedDom(target_number_of_sensors = target_number_of_sensors, verbose = False)
    out_flag = RDOED.out_flag
    if out_flag[2] != "Failed":
        print("Succeeded for " + str(target_number_of_sensors).zfill(5) + " with flag " + str(out_flag))
        wrd = RDOED.design
    else:
        print("Failed for " + str(target_number_of_sensors).zfill(5) + " with flag " + str(out_flag))
        wrd = None
    J_init.fac = prefac
    objective = J_init.eval(wrd)
    J_init.fac = 1
    return objective, out_flag

#def do_objective(w):
#    print("Computing objective for " + str(round(np.sum(w))) + "...")
#    return 

def random_design(target_number_of_sensors, tries = 100):
    objective = np.inf
    for _ in range(tries):
        w = np.zeros(m)
        w[rng.choice(m, size = target_number_of_sensors, replace = False)] = 1
        objective = np.min([objective, J_init.eval(w)])
    print("Random design for " + str(target_number_of_sensors) + " chosen.")
    return objective

target_filename = str(m).zfill(5)
load_flag = False
for filename in listdir(tests_folder):
    if isfile(join(tests_folder, filename)):
        if target_filename in filename and not ".png" in filename:
            with open(tests_folder + filename, "rb") as input_file:
                obj = pickle.load(input_file)
       
                #ws = obj["ws"]
                out_flags = obj["out_flags"]
                
                objectives = obj["objectives"]
                random_objectives = obj["random_objectives"]
                
                load_flag = True
                break
                
if load_flag:
    print("Loaded!")
    targets = np.arange(m+1)
    plt.figure(figsize=(24,12))
    plt.plot(targets,objectives,'b')
    plt.plot(targets,random_objectives,'r')
    plt.savefig(tests_folder + "Objectives" + str(m).zfill(5) + ".png")

if not load_flag:
    filename = target_filename
     
    objectives = []
    out_flags = []   
    
    #for target in range(m):
    #    out = testRedDom(target)
    #    ws.append(out[0])
    #    out_flags.append(out[1])
    if __name__ == '__main__':
        with Pool(processes=int(cpu_count()*3/4)) as pool:
            out = pool.map(testRedDom, range(m+1))

    
    for outk in out:
        objectives.append(outk[0])
        out_flags.append(outk[1])

    #if __name__ == '__main__':
    #    with Pool(processes=int(cpu_count()*3/4)) as pool:
    #        objectives = pool.map(do_objective, ws)

    rng = np.random.default_rng(311)
    
    J_init.fac = prefac
    if __name__ == '__main__':
        with Pool(processes=int(cpu_count()*3/4)) as pool:
            random_objectives = pool.map(random_design, range(m+1))
    J_init.fac = 1
    
    targets = np.arange(m+1)
    plt.figure(figsize=(24,12))
    plt.plot(targets,objectives,'b')
    plt.plot(targets,random_objectives,'r')
    plt.savefig(tests_folder + "Objectives" + str(m).zfill(5) + ".png")
            
    obj = {"out_flags": out_flags, \
           "objectives": objectives, "random_objectives": random_objectives}
    
    with open(tests_folder + filename, "wb") as output_file:
        pickle.dump(obj, output_file)    
    print("Successfully tested with m = " + str(m) + ".")
