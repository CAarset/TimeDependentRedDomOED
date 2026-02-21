from ngsolve import *
from netgen.occ import *
import numpy as np
import dill as pickle
from itertools import count
from safe_add_CFs import safe_add_CFs, interp_add_CFs
from scipy.sparse import lil_matrix

from os import listdir, makedirs
from os.path import isfile, join
from time import time
from datetime import timedelta

class grid_maker():
    def __init__(self, mmaker, target_m = 100, \
        integration_order = 10, solver = '', domain_is_complex = False, shape = "pointwise", sensor_radius = None, **kwargs):

        self.mmaker = mmaker
        self.mesh = mmaker.mesh
        
        self.shape = shape

        self.integration_order = integration_order
        
        if sensor_radius is None:
            self.sensor_radius_square = None
            sensor_radius_name = str(0)
        else:
            self.sensor_radius_square = sensor_radius**2
            sensor_radius_name = str(round(sensor_radius,5))
        self.base = 24
        
        self.cofes = mmaker.cofes
        
        self.dtype = np.float32
            
        # Complex data is treated as double-length real vectors
        self.out_map = lambda g: g.real
        self.out_map_inv = lambda g: g.real

        grid_dumps_directory = "grid_dumps/"
        makedirs(grid_dumps_directory, exist_ok = True)
        
        target_filename = "target_" + str(target_m) + "_shape_" + shape + "_sensorradius_" + sensor_radius_name + "_" + self.mmaker.target_filename
        self.target_filename = target_filename 
        
        load_flag = False
        # Attempt to find a stored grid with the same target
        for filename in listdir(grid_dumps_directory):
            if isfile(join(grid_dumps_directory, filename)):
                if target_filename in filename:
                    with open(grid_dumps_directory + filename, "rb") as input_file:
                        obj = pickle.load(input_file)

                        self.grid = obj["grid"]
                        self.sqrtm = obj["sqrtm"]
                        self.base = obj["base"]
                        self.Omat = obj["Omat"]
                        self.obstime = obj["obstime"]
                        self.circ_coords = obj["circ_coords"]
                        
                        load_flag = True
                        print("Loaded stored grid " + filename + "...")
                        break

        if not load_flag:
            print("No saved grid found, generating...")
            filename = grid_dumps_directory + target_filename
                        
            # Create observation grid
            
            self.sqrtm = int(np.log2(target_m/self.base + 1)) #np.ceil(np.sqrt(target_m)).astype(int)
            while True:
                X = np.linspace(-0.2,1,self.sqrtm)
                Y = np.linspace(-1,1,self.sqrtm)
                X, Y = np.meshgrid(X,Y)

                coords = np.stack((X.ravel(), Y.ravel()), axis = 1)
                off = 0.05
                in_mesh = np.ones_like(coords[:,0])
                for offset in [[0,0],[-off,0],[off,0],[0,-off],[0,off]]:
                    offset = np.array(offset)
                    in_mesh = np.logical_and(in_mesh,np.array([self.mesh(*(x+offset)).nr > -1 for x in coords],dtype=bool))
                if np.sum(in_mesh) < target_m:
                    self.sqrtm += 1
                    continue
                else:
                    coords = coords[in_mesh,:]
                    del X, Y, in_mesh #x_norms, 
                    break
        
            self.grid = coords
            self.circ_coords = self.grid[np.argsort(-np.linalg.norm(coords, axis = 1)),:]
            print("Finished building grid.")
            
        # Count the number of sensors.
        # For complex data, we treat it as though every sensor appears twice,
        # with two outputs "in the same spot" - 
        # one real, one complex, concatenated as a real double-length vector.
        self.m_sensors = self.grid.shape[0]
        self.n = self.mmaker.n
            
        # Create observation functions
        
        from ngsolve import x, y

        if self.sensor_radius_square is None:
            self.sensor_radius_square = 0.95 / (max(25,self.sqrtm) ** 2)
        
        if shape.casefold() == "round".casefold():
            def create_observation(coordx, coordy):
                return CoefficientFunction(\
                        IfPos(self.sensor_radius_square - ((x - coordx)**2 + (y - coordy)**2), \
                              self.sensor_radius_square - ((x - coordx)**2 + (y - coordy)**2),0))
            self.peak = self.sensor_radius_square
        
        elif shape.casefold() == "pointwise".casefold():
        
            def create_observation(coordx, coordy):
                r2 = (x - coordx)**2 + (y - coordy)**2
                return CoefficientFunction(\
                       exp(-3 / self.sensor_radius_square * r2))
            self.peak = 1
        
        elif shape.casefold() == "gauss".casefold():
             
            moll = lambda R: exp(-1/(1 - R))
            cut = lambda R: IfPos(1 - R, moll(R), 0)
            
            def create_observation(coordx, coordy):
                r2 = (x - coordx)**2 + (y - coordy)**2
                return CoefficientFunction(\
                         cut(r2/self.sensor_radius_square))
            self.peak = np.exp(-1)
            
        elif shape.casefold() == "hat".casefold():
             
            def create_observation(coordx, coordy):
                 return CoefficientFunction(\
                        IfPos(self.sensor_radius_square - ((x - coordx)**2 + (y - coordy)**2),1,0))
            self.peak = 1
            
        else:
            raise ValueError("Unknown sensor shape",shape,"accepted values are round, gauss and hat (case insensitive).")
        
        self.create_observation = create_observation
        observation_function = create_observation(coordx = 0, coordy = 0)

        if shape.casefold() != "pointwise".casefold():
            self.norm = np.real(Integrate(observation_function*Conj(observation_function), self.mesh, order = self.integration_order))**(1/2)
        else:
            self.norm = 1
        
        assert self.norm > 1e-20, "Observation function norm too small (" + str(self.norm) + "), alter definition!"
        self.peak /= self.norm
        
        # Observation functions defined as a generator for memory purposes
        def observation_functions():
            for coordx, coordy in self.grid:
                yield create_observation(coordx = coordx, coordy = coordy) / self.norm
                    
        self.observation_functions = observation_functions
        
        # Save output
        if not load_flag:
            
            # Create sparse observation matrix for efficiency.
            cofes = self.mmaker.cofes
            V = cofes.TrialFunction()
            n_cofes = cofes.ndof

            print("Building observation matrix, this may take some time...")
            self.Omat = lil_matrix((self.m_sensors,n_cofes))
            
            start_obstime = time()

            if not self.shape.casefold() == "pointwise".casefold():
                obs = self.observation_functions()
                
            for k, coord in enumerate(self.grid):
                print("\r","Building observation matrix: ",k,"/",self.m_sensors,"...",sep="",end="")

                # This gives precisely the FEM's interpretation of
                # pointwise evaluation by storing each basis element's
                # value in each sensor point. This can safely be stored
                # as a sparse matrix, since the (local) basis will only
                # have very few elements touching each sensor
                
                Lo = LinearForm(cofes)
                if self.shape.casefold() == "pointwise".casefold():
                    Lo += V(*coord)
                else:
                    Lo += next(obs) * V
                Lo.Assemble()
                Lo = Lo.vec.FV().NumPy()[:]
                assert np.all(Lo.imag == 0), "This approach does not work with complex-valued bases!"
                
                self.Omat[k,:] = Lo.real
                
            self.Omat = self.Omat.tocsc()            
            self.obstime = time() - start_obstime
            print("\nBuilt observation matrix in",timedelta(seconds = self.obstime))
            
            obj = {"grid": self.grid, "sqrtm": self.sqrtm, "base": self.base, "Omat": self.Omat, \
                   "circ_coords": self.circ_coords, "obstime": self.obstime}
            with open(filename, "wb") as output_file:
                pickle.dump(obj, output_file)    
            print("Successfully stored mesh, grid & observation functions with m_sensors = " + str(self.m_sensors) + ".")

    def grid_to_ngs(self, w, interp = False, fes = None):
        # Used to convert design vector w to ngs function for observation.
        if interp:
            if fes is None:
                fes = self.mmaker.designfes
            gf = interp_add_CFs(CFs = self.observation_functions(), weights = w, fes = fes)
            return gf
        else:
            cf = safe_add_CFs(CFs = self.observation_functions(), weights = w)
            return cf

    def Obs(self, u):
        shape = self.shape
        if self.shape.casefold() == "pointwise".casefold():
            with TaskManager():
                g = u(self.mesh(self.grid[:,0],self.grid[:,1])).ravel()
        else:
            with TaskManager():
                g = np.array( \
                    Integrate(tuple(obs * u for obs in self.observation_functions()), self.mesh)).ravel()
        return g
    
    def O(self, u):
        with TaskManager():
            g = self.Omat@self.mmaker.ngs_to_coeff(u)
        return self.out_map(g)

    def OT(self, g):
        cofes = self.mmaker.cofes
        with TaskManager():
            u = GridFunction(cofes)
            u.vec.FV().NumPy()[:] = self.Omat.T.conj()@self.out_map_inv(g).conj()
            v = GridFunction(cofes)
            v.vec.data = cofes.MassMatrix.Inverse(freedofs = cofes.FreeDofs()) * u.vec
        return v

    def F(self, f, obs_times = [1]):
        g = np.array([],dtype=self.dtype)
        
        t = 0
        u = 0
        for obs_time in obs_times:
            u = self.mmaker.A(f = f, t0 = t, t1 = obs_time)
            t = obs_time
            g = np.concatenate((g,self.O(u)))
        return g

    def FT(self, g, obs_times = [1]):
        
        f = GridFunction(self.mmaker.fes)
        
        t = obs_times[-1]
        u = self.OT(g)
        
        f = self.mmaker.AT(u, t0 = 0, t1 = t)
        return f

    def FC(self, r, obs_times = [1]):
        f = self.mmaker.real_to_ngs(r)
        return self.F(self.mmaker.C(f), obs_times = obs_times)

    def CFT(self, g, obs_times = [1]):
        f = self.mmaker.C(self.FT(g, obs_times = obs_times))
        return self.mmaker.ngs_to_real(f)

