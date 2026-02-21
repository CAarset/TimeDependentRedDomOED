from ngsolve import *
from ngsolve.webgui import Draw
from ngsolve.internal import SnapShot

mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))

cf = cos(x + 3*y) - sin(2*x - y**2)
help(SnapShot)
if 1:
    Draw(cf, mesh)
    SnapShot(filename = "test.png")
    SnapShot(filename = "test.bmp")
    SnapShot(filename = "test")
else:
    fes = L2(mesh, order = 2)
    f = GridFunction(fes)
    f.Set(cf)

    Draw(f, mesh)
    SnapShot(filename = "test2.png")
    SnapShot(filename = "test2.bmp")
    SnapShot(filename = "test2")