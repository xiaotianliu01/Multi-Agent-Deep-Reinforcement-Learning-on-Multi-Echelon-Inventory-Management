import numpy as np
import random
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class merton(object):

    def __init__(self, length, max_demand = 20):

        base = int(max_demand/2)
        start = 0
        var = 15
        delta = 0.01
        delta_t = var
        u = 0.5*delta*delta
        a = 0
        b = 0.01
        lamda = var
        
        while(True):
            self.demand_list = []
            self.drump = []
            self.no_drump = []
            self.no_drump.append(start)
            self.demand_list.append(start)
            self.drump.append(0)
            for i in range(length):
                Z = np.random.normal(0, 1)
                N = np.random.poisson(lamda)
                Z_2 = np.random.normal(0, 2)
                M = a*N + b*(N**0.5)*Z_2
                new_X = self.demand_list[-1] + u - 0.5*delta*delta + (delta_t**0.5)*delta*Z + M
                self.demand_list.append(new_X)
            self.demand_list = [int(math.exp(i)*base) for i in self.demand_list]
            if(np.mean(self.demand_list)>0 and np.mean(self.demand_list)<max_demand):
                break
            
        for i in range(len(self.demand_list)):
            self.demand_list[i] = min(max_demand, self.demand_list[i])  

    def __getitem__(self, key = 0):
        return self.demand_list[key]

class stationary_possion(object):

    def __init__(self, length, max_demand=20):
        self.demand_list = np.random.poisson(max_demand/2, length)
    
    def __getitem__(self, key = 0):
        return self.demand_list[key]
