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
delta_t = 1e-3

rng = np.random.default_rng()
robin = 1
alpha = 0.001

target_m = int(4e2)
maxh = 0.03

clear = False

RDOED, RNGOED = runOED(
           target_m = target_m, integration_order = 5, 
           maxh = maxh, order = 2,
           tol = 1e-15,
           t1 = t1, delta_t = delta_t,
           robin = robin, alpha = alpha,
           sensor_radius = None, verbose = False,
           clear = clear)

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
def pretty(f,u=None,w=None,maxval=None):
    F = GridFunction(mmaker.designfes)
    F.Set(f)

    if maxval is None:
        maxval = ngs_to_max(F,F.space.mesh)

    FF = GridFunction(mmaker.codesignfes)
    if w is None and u is None:
        FF.Set(F)
    else:
        if u is not None:
            FF.Set(F + u)
        else:
            FF.Set(F + gmaker.grid_to_ngs(w) * maxval / gmaker.peak)
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
    priorfield = mmaker.fieldPrior
    c0 = ngs_to_max(priorfield, mesh = mesh)

    # Dump a graphic of the prior field
    Draw(pretty(priorfield), min = 0, max = c0);
    Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/" + priorname)
    
    Draw(mesh);
    Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/" + meshname)
    
    # Max saved to a csv for TeX colorbar scaling
    design.append("zero")
    posteriormax.append(c0)
    
    # Pointwise variance of full-sensors cov
    covfield = design_to_field(ones)
    c1 = ngs_to_max(covfield,dmesh)
    
    # Posterior max will be saved to a csv for TeX colorbar scaling
    design.append("ones")
    posteriormax.append(c1)
    
    posteriormax = ['{:.18f}'.format(c0) for c0 in posteriormax]
    np.savetxt(paper_graphics_directory + "/" + posteriorname, np.column_stack((design,posteriormax)), header = header, comments = "", fmt = "%s", **kwargs)
    
# Fictional source f, used for examples
f = GridFunction(fes)
f.Load("f")

t1 = RDOED.obs_times[0]

if 1:
        u = A(f)
        U = GridFunction(dddfes)
        U.Set(np.real(u))

        F = GridFunction(ddfes)
        F.Set(f)

        U0 = ngs_to_max(U,dmesh)

        UU = GridFunction(dddfes)
        UU.Set(buff(U, a = -0.31, b = 0.31, A = -U0, B = U0))

        f_u = GridFunction(dfes)
        f_u.Set(F + UU) # The rescaling on UU makes it look nicer and doesn't really affect any results

        FF = GridFunction(dfes)
        FF.Set(F)

        ########################################
        ############### Figure ? ###############
        ########## ? ##########
        ########################################

        #ngsint.visoptions.numtexturecols = 16
        Draw(pretty(clamp(f_u,0,1)), "f_u", min=0, max=1);
        sleep(1)
        Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/f_and_u.ppm")
        #ngsint.visoptions.numtexturecols = 256

########################################
############### Figure ? ###############
### Sensor grid and covariance field ###
##### corresponding to all sensors ##### 
########################################

gridname = "grid.ppm"
if gridname in listdir(paper_graphics_directory):
    pass
else:
    # Smaller sensors to avoid overcrowded outer ring
    #gmaker.sensor_radius_square /= 6
    
    wcf = gmaker.grid_to_ngs(ones) * c1 / gmaker.peak # Max of sensors should match max of colorbar
    
    covff.Set(design_to_field(ones))
    
    covf.Set(covff)
    covf2.Set(wcf)
    covf.vec.data += covf2.vec.data
      
    Draw(covf, min = 0, max = c0);
    sleep(1)
    Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/" + gridname)

    # Return sensor size to normal
    #gmaker.sensor_radius_square *= 6

########################################
############### Figure 4 ###############
##### Study of global optimum with #####
### emphasis on optimality criteria ####
########################################

target = 36
w1 = w1s[target]

w1, j = RDOED.set_binary(w = w1, target = target, return_jac = True, sort = True, set_red_dom = True)
doms = RDOED.dom_indices 
free = RDOED.free_indices
reds = RDOED.red_indices

doms = np.concatenate((doms,[free[0]]))
free = np.concatenate((free,[reds[0]]))

# Note the +1 index offsets to match with mathematical indexing from 1
np.savetxt(paper_graphics_directory + "/w1_36_dom.csv",np.column_stack((w1[doms],j[doms],doms+1)), delimiter=",", header="w,jac,ind",comments="")
np.savetxt(paper_graphics_directory + "/w1_36_free.csv",np.column_stack((w1[free],j[free],free+1)), delimiter=",", header="w,jac,ind",comments="")
np.savetxt(paper_graphics_directory + "/w1_36_red.csv",np.column_stack((w1[reds],j[reds],reds+1)), delimiter=",", header="w,jac,ind",comments="")

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
#### p-relaxed designs for m0 = 24 #####
########################################

target = 36

w1 = w1s[target]
w1, jac = RDOED.set_binary(w = w1, target = target, return_jac = True, sort = True, set_red_dom = True) # Prints to console number of red/doms
order = np.argsort(jac)

wseq = wseqs[target]
pseq = ps[target]

wseq.insert(0, w1)
pseq.insert(0, 1)

#subseq_choice = np.array(np.array([1,2,3,5,7,9,11,13,14]) - 1).astype(int)

#pseq = [pseq[choice] for choice in subseq_choice]
#wseq = [wseq[choice] for choice in subseq_choice]

header = "index,"
np.savetxt(paper_graphics_directory + "/pseq_" + str(target) + ".csv",np.column_stack((np.arange(len(pseq)),pseq)), fmt = "%i, %.3f", delimiter=",", header="step,p",comments="")
allws = np.arange(len(wseq[0]))+1 # Index array, offset by one to match mathematical indexing starting at 1

for i, w in enumerate(wseq):
    allws = np.column_stack((allws,np.fmin(1,np.fmax(0,w))[order])) # Sorts and prevents numerical rounding issues
    header += str(i+1) # The usual offset by 1
    if i < len(wseq) - 1:
        header += ","
np.savetxt(paper_graphics_directory + "/wseq_24.csv",allws, delimiter=",", header=header,comments="")

########################################
############### Figure ? ###############
#### p-relaxed designs for m0 = 24 #####
########################################

cov0name = "cov_w1_"
covname = "cov_w_"
covRNGname = "cov_wRNG_"

maxfieldPrior = ngs_to_max(mmaker.fieldPrior,mesh)

for target in np.arange(0,37):
    if not target:
        Draw(pretty(mmaker.fieldPrior),"cov", min = 0, max = maxfieldPrior);
        sleep(1)
        Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/priorfield.ppm")
        continue
    w = ws[target]
    w1 = w1s[target]
    wRNG = wRNGs[target]

    for W, name in zip([w1, w, wRNG],[cov0name, covname, covRNGname]):
        name += str(target).zfill(3)
        try:
            with open(os.path.join(paper_graphics_directory,name), "rb") as input_file:
                covfield = pickle.load(input_file)
        except:
            print("\n")
            print("No covfield found for " + name + ", building...",sep="")
            covfield = design_to_field(W)
            with open(os.path.join(paper_graphics_directory,name), "wb") as output_file:
                pickle.dump(covfield, output_file)
            
        name += ".ppm"
            
        #if not i:
        #    covfield0 = covfield
        #    cw1 = ngs_to_max(covfield,dmesh)
        #    cw1 = cw0
        #    covff.Set(covfield)
        #else:
        #    if i == 1:
        #        covff.Set(covfield - covfield0)
        #        cw1 = 1.8 * ngs_to_max(covff, dmesh)
        #        np.savetxt(paper_graphics_directory + "/posterior_24.csv", np.column_stack((["24","24diff"],[cw0,cw1])), header = "design,posteriormax", comments = "", fmt = "%s", **kwargs)
        #    covff.Set(clamp(covfield - covfield0,0,cw1))
                
        wcf = gmaker.grid_to_ngs(W) * maxfieldPrior / gmaker.peak # Max of sensors should match max of colorbar
        
        Draw(pretty(covfield,w=W,maxval=maxfieldPrior),"cov", min = 0, max = maxfieldPrior);
        sleep(1)
        Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/" + name)

        #i += 1
        
########################################
############### Figure ? ###############
##### Reconstructions for m0 = 24 ######
########################################

target = 24
w = ws[target]
w1 = w1s[target]

Draw(pretty(FF), min = 0, max = 1);
sleep(1)
Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/f.ppm")

reco_w = RDOED.design_to_sol(w = w, f = cf)

Draw(pretty(reco_w,w=w), min = 0, max = 1)
sleep(1)
Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/f_reco_24.ppm")

Draw(reco_w-F,w=w, min = 0, max = 1)
sleep(1)
Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/f_reco_error_24.ppm")

########################################
############### Figure ? ###############
###### Pointwise covariance field ######
########################################

# Limit to 24 designs (no significant further change to variance field after this point)
final_target = 36
final_target_digits = int(np.log10(final_target)+1)

covf = GridFunction(dfes)

# c0 refers to max of prior field
c0 = transform(c0)

for target in range(1,final_target+1):

    snapshot = str(target).zfill(final_target_digits)
    covname = "cov" + snapshot
    # Skip if already exists (especially convenient if multiple runs are needed
    #if covname in listdir(paper_graphics_directory):
    #    continue
        
    w_test = ws[target]

    try:
        with open(os.path.join(paper_graphics_directory,covname), "rb") as input_file:
            covfield = pickle.load(input_file)
    except:
        print("No covfield found for " + covname + ", building...",sep="")
        covfield = design_to_field(w_test)
        with open(os.path.join(paper_graphics_directory,covname), "wb") as output_file:
            pickle.dump(covfield, output_file)
            
    covname += ".ppm"
    
    covff.Set(transform(covfield))
    wcf = gmaker.grid_to_ngs(w_test) * c0 / gmaker.peak # Max of sensors should match max of colorbar
    covf.Set(covff + wcf)
        
    Draw(covf, min = 0, max = c0);
    sleep(1)
    Snapshot(w = image_width, h = image_height, filename = paper_graphics_directory + "/" + covname)
