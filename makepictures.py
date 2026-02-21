from os import listdir
from os.path import isfile, join
from vtk_to_png import *

for f in listdir(mypath) if isfile(join(mypath, f)):
    vtk_to_png(filename = f, \
               imagename = f.split(sep=".")[0] + ".png", \
               delete = True)