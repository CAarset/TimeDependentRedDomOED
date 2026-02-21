from OEDAlgorithm import *
import numpy as np
from itertools import permutations

class GreedyOED(OEDAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.greedy_sequence = []
        self.target_number_of_sensors = self.m

    def set_w(self, w, target_index):
        wt = deepcopy(w)
        assert np.all(wt[target_index] == 0), "Targeting non-zero sensor!"
        wt[target_index] = 1
        return wt
        
    def step(self, w, target_index):
        
        wt = self.set_w(w, target_index)

        if self.test_optimality(wt, full = False, exit = False):
            return wt, True
        else:
            return wt, False

    def Greedy(self, target_number_of_sensors = None, verbose = False):
       
        self.verbose = verbose
        
        # Target number can be set freely, all cases use the decomposition computed in init.
        if target_number_of_sensors is None:
            target_number_of_sensors = self.m
        else:
            self.target_number_of_sensors = target_number_of_sensors
        
        # Reset free indices for this test.
        self.free_indices = np.arange(self.m, dtype = int)
        self.dom_indices = np.array([], dtype = int)
        self.red_indices = np.array([], dtype = int)
        self.update(dominant_indices = np.array([], dtype = int), redundant_indices = np.array([], dtype = int))
        
        self.Bh0 = deepcopy(self.Bh)
        w = np.zeros(self.m, dtype = int)
        
        self.current_target = 0
        current_target = 0
        extra = 0
        self.successes = []
        self.global_optimality = []
        
        while current_target < self.target_number_of_sensors:
            
            current_target += 1
            self.current_target = 1 + extra
            
            J = self.Jac(w)
            target_indices = np.flip(np.argsort(-J))
            
            for i, target_index in enumerate(target_indices): #permutations(target_indices, extra + 1):

                #target_index = np.array(target_index)
                actual_target_index = self.free_indices[target_index]
                print("\r" + str((i,actual_target_index)), end = "")
                wt, flag = self.step(w, actual_target_index)
                if flag:
                    self.successes.append(current_target)
                    obj_val = self.J_init.eval(wt)
                    break
            
            if not flag:
                i = np.inf
                #print("Failed for",current_target)
                obj_val = np.inf
                for target_index in self.free_indices:
                    wt = self.set_w(w, target_index)
                    obj_val_t = self.J_init.eval(wt)
                    if obj_val_t < obj_val:
                        obj_val = obj_val_t
                        best_index = target_index
                actual_target_index = best_index
                target_index = np.argwhere(self.free_indices==actual_target_index)[0]
                self.successes.append(-actual_target_index)
            
            w = self.set_w(w, actual_target_index)
            self.update(dominant_indices = np.array([actual_target_index], dtype = int).ravel(), \
                        redundant_indices = np.array([], dtype = int))
            glob = self.test_optimality(w, full = True, exit = False)
            
            self.greedy_sequence.append(actual_target_index)
            self.global_optimality.append(glob)
            
            print(np.sum(wt),actual_target_index,J[target_index],obj_val,flag,glob,i)
            
                #extra += 1
                #raise Exception("Didn't work, got to" + str(self.current_target) + "...")
                
    def greedy_design(self, target):
        w = np.zeros(self.m)
        w[np.array(self.greedy_sequence[:target],dtype=int)] = 1
        return w
            