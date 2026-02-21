from ngsolve import *
from netgen.gui import Snapshot
import ngsolve.internal as ngsint
from os import mkdir
from copy import deepcopy
import dill as pickle
from util import *
import matplotlib.pyplot as plt
from time import time, sleep
from os import makedirs, listdir
from runOEDmethod import *

t1 = 1
delta_t = 1e-4

rng = np.random.default_rng()

robin = 1
alpha = 0.25
scale = 2

target_m = int(4e2)
maxh = 0.03
refine_inner = 0

clear = False

RDOED, RNGOED = runOED(
           target_m = target_m, integration_order = 5, 
           maxh = maxh, order = 2, refine_inner = refine_inner,
           tol = 1e-15,
           t1 = t1, delta_t = delta_t,
           robin = robin, alpha = alpha, scale = scale,
           sensor_radius = None, verbose = False,
           clear = clear, compute_all = False)

# Setup, also called by runOED
n, m, m_sensors, m_obs = RDOED.n, RDOED.m, RDOED.m_sensors, RDOED.m_obs

ms = m_sensors
ones = np.ones(ms)
zero = np.zeros(ms)

mmaker = RDOED.mmaker
gmaker = RDOED.gmaker
gmaker.sensor_radius_square *= 1/2

rng = mmaker.rng

mesh = mmaker.mesh
fes = mmaker.fes
cofes = mmaker.cofes

eva = RDOED.J_init.eval
jac = RDOED.J_init.jac

A = RDOED.A
AT = RDOED.AT
F = RDOED.F
FT = RDOED.FT

# Save everything to the paper graphics folder
paper_graphics_directory = "graphics/"
os.makedirs(paper_graphics_directory, exist_ok = True)

# Resolution for 4K monitors
image_width = 3840
image_height = 2160

# Load precomputed p-relaxed designs
fn = RDOED.output_filename
with open(fn, "rb") as filename:
    obj = pickle.load(filename)

# Load the A-optimality of 10^3 random 
# designs for each target number of sensors
fnRNG = fn + "_RNG"
with open(fnRNG, "rb") as filename:
    objRNG = pickle.load(filename)

# Dicts organised by target number
ws = obj["ws"]
w1s = obj["w1s"] # Actually globally optimal non-binary designs (i.e. 1-relaxed)
ps = obj["ps"]
wseqs = obj["wseqs"]

wRNGs = objRNG["ws"]

allvals = objRNG["allvals"]

# Log transform to keep later variances visible
transform = lambda cf: cf#log(1 + cf)

# Pushes some values deliberately outside of clamp range to emphasise visually
def buff(cf, a, b, A = 0, B = 1):
    bcf = (cf - A) / (B - A)
    bcf = a + (b - a) * bcf
    return bcf
    
# clamp function for when ngsolve fails to respect min and max (why?)
clamp = lambda cf, low, high: IfPos(high-cf,IfPos(cf,cf,low),high)

# norm function
order = 15
norm = lambda f: Integrate(f**2,mesh.Materials("src"),order=order)**(1/2)

# Prettyfying function for drawing
def pretty(f,co=False,u=None,w=None,maxval=None):
    print("Pretty1")
    if co:
        F = GridFunction(mmaker.codesignfes)
    else:
        F = GridFunction(mmaker.designfes)
    F.Set(f)
    print("Pretty2")
    
    if maxval is None:
        maxval = ngs_to_max(F,mesh)
    print("Pretty3")
    
    FF = GridFunction(mmaker.codesignfes)
    if w is None and u is None:
        FF.Set(F)
    else:
        if u is not None:
            FF.Set(F + u)
        else:
            FF.Set(F + gmaker.grid_to_ngs(w) * maxval / gmaker.peak)
    print("Pretty4")
    return FF

# Removes mesh grid, colorbar, logo, xyz-axis marker and sets to a much finer color scale than ngsolve default
ngsint.viewoptions.drawoutline=0
ngsint.viewoptions.drawcolorbar=0
ngsint.viewoptions.drawnetgenlogo=0
ngsint.viewoptions.drawcoordinatecross=0
ngsint.visoptions.numtexturecols = 1024

########################################
############ Precomputation ############
###### Prior (no observation) & ########
## full observation covariance fields ##
########################################

# Skip if already exists (especially convenient if multiple runs are needed
meshname = "mesh.ppm"
posteriorname = "posteriors.csv"
priorname = "cov000.ppm"

header = "design,posteriormax"
delimiter = ","
kwargs = {
    "delimiter": delimiter,
}

design_to_field = RDOED.design_to_field

# Utility wrapper for design-to-cov

if meshname in listdir(paper_graphics_directory) and \
   posteriorname in listdir(paper_graphics_directory) and \
   priorname in listdir(paper_graphics_directory):
    design, posteriormax = np.genfromtxt(paper_graphics_directory + "/" + posteriorname, dtype = "str,float", unpack=True,**kwargs)
    c0 = posteriormax[1] 
else:

    # Dumps for .csv saving
    design = []
    posteriormax = []
    
    # Pointwise variance of prior cov
    priorfield = mmaker.diagPrior
    c0 = ngs_to_max(priorfield, mesh = mesh)

    # Dump a graphic of the prior field
    Draw(priorfield, mesh, "priorfield", min = 0, max = c0);
    Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/" + priorname)
    
    Draw(mesh);
    Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/" + meshname)
    
    # Max saved to a csv for TeX colorbar scaling
    design.append("zero")
    posteriormax.append(c0)
    
    # Pointwise variance of full-sensors cov
    covfield = design_to_field(ones)
    c1 = ngs_to_max(covfield,mesh)
    
    # Posterior max will be saved to a csv for TeX colorbar scaling
    design.append("ones")
    posteriormax.append(c1)
    
    posteriormax = ['{:.18f}'.format(c0) for c0 in posteriormax]
    np.savetxt(paper_graphics_directory + "/" + posteriorname, np.column_stack((design,posteriormax)), header = header, comments = "", fmt = "%s", **kwargs)
    
# Fictional source f, used for examples
f = GridFunction(fes)
f.Load("f")

u = GridFunction(cofes)
u.Load("u")


#ngsint.visoptions.numtexturecols = 16
Draw(pretty(clamp(f,0,1e3)), min=0, max=1e3);
sleep(1)
Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/s.ppm")

Draw(pretty(u, co = True), min=0);
sleep(1)
Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/u.ppm")
#ngsint.visoptions.numtexturecols = 256

########################################
############### Figure ? ###############
### Sensor grid and covariance field ###
##### corresponding to all sensors ##### 
########################################

gridname = "grid.ppm"

nil = GridFunction(cofes)

Draw(pretty(nil, w = ones, maxval = gmaker.peak), min = 0, max = gmaker.peak);
sleep(1)
Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/" + gridname)

    # Return sensor size to normal
    #gmaker.sensor_radius_square *= 6

########################################
############### Figure ? ###############
########### Comparison plots ###########
########################################

targets = []
w1vals = []
wvals = []
minvals = []
maxvals = []

upper_graphics_limit = np.inf # Do not plot anything above this value to keep the plot clean
header = "targets,w1,w,randommin,randommax"
for target in range(1,37):
    targets.append(target)
    w1vals.append(eva(w1s[target]))
    wvals.append(eva(ws[target]))
    minvals.append(min(upper_graphics_limit,np.min(allvals[target])))
    maxvals.append(min(upper_graphics_limit,np.max(allvals[target])))
    
np.savetxt(paper_graphics_directory + "/Aoptimalities.csv", np.column_stack((targets,w1vals,wvals,minvals,maxvals)), delimiter=",", header=header, comments="")

########################################
############### Figure ? ###############
##### Reconstructions for m0 = 36 ######
########################################

fmax = 1e3

target = 36
w = ws[target]

fr = GridFunction(fes)
fr.Load("fr_OED")

Draw(pretty(fr,w=w,maxval=fmax), min = 0, max = fmax)
sleep(1)
Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/f_reco_36.ppm")
sleep(1)

frRNG = GridFunction(fes)
frRNG.Load("fr_RNG")

Draw(pretty(frRNG,w=w,maxval=fmax), min = 0, max = fmax)
sleep(1)
Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/fRNG_reco_36.ppm")
sleep(1)

########################################
############### Figure ? ###############
########## Field for m0 = 36 ###########
########################################

target = 36
w = ws[target]
    
maxfieldPrior = ngs_to_max(mmaker.diagPrior,mesh)

field = GridFunction(fes)
field.Load("field")

Draw(pretty(field,w=w,maxval=maxfieldPrior),"cov", min = 0, max = maxfieldPrior);

sleep(1)
Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/field.ppm")
sleep(1)
