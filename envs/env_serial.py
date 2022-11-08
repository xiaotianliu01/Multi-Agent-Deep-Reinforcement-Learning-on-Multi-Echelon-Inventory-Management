import numpy as np
from . import generator
import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

C = [1, 1, 1, 1, 1]
DISCOUNT = [1, 0.95, 0.9, 0.85, 0.8]
H = [1, 1, 1, 1, 1]
B = [1, 1, 1, 1, 1]
PR = [4, 4, 4, 4, 4]
S_I = 10
S_O = 10
LEAD_TIME = 4
LEVEL_NUM = 3
ACTION_DIM = 21
OBS_DIM = LEAD_TIME + 3
ALPHA = 0.5
EVAL_PTH = "F:\\test_data\\"
RSL = False
LOST_RATE = 0.1
EPOSIDE_LEN = 200

def get_eval_data():
    eval_files = os.listdir(EVAL_PTH)
    eval_data = []
    n_eval = len(eval_files)
    for file in eval_files:
        data = []
        with open(EVAL_PTH + file, "rb") as f:
            lines = f.readlines()
            for line in lines:
                data.append(int(line))
        eval_data.append(data)
    return n_eval, eval_data

class Env(object):

    def __init__(self):
        self.agent_num = LEVEL_NUM
        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM 
        self.r_s_l = RSL
        self.inventory = []
        self.backlog = []
        self.order = []
        self.record_act_sta = [[] for i in range(LEVEL_NUM)]

        self.eval_eposide_len = EPOSIDE_LEN
        self.eposide_max_steps = EPOSIDE_LEN

        self.n_eval, self.eval_data = get_eval_data()
        self.eval_index = 0

    def reset(self, train = True, normalize = True):

        self.backlog = [0 for i in range(LEVEL_NUM)] 
        self.action_history = [[] for i in range(LEVEL_NUM)]
        self.train = train
        self.normalize = normalize
        self.step_num = 0
        self.level_num = LEVEL_NUM
        self.inventory = [S_I for i in range(LEVEL_NUM)]
        self.order = [[S_O for i in range(LEAD_TIME)] for j in range(LEVEL_NUM)]

        if(train == True):
            self.demand_list = generator.merton(EPOSIDE_LEN, ACTION_DIM-1)
        else:
            self.demand_list = self.eval_data[self.eval_index]
            self.eval_index += 1
            if(self.eval_index == self.n_eval):
                self.eval_index = 0

        sub_agent_obs = []
        for i in range(self.level_num):
            if(normalize):
                arr = np.array([self.inventory[i], 0, S_O] + self.order[i])/(ACTION_DIM-1)
            else:
                arr = np.array([self.inventory[i], 0, S_O] + self.order[i])
            arr = np.reshape(arr, (self.obs_dim,))
            sub_agent_obs.append(arr)
            
        return sub_agent_obs

    def step(self, actions, one_hot = True):
        
        if(one_hot):
            action_ = [np.argmax(i) for i in actions]
        else:
            action_ = actions
        self.current_orders = action_
        action = [i for i in action_]
        reward = self.state_update(action)
        
        state = []
        if(self.normalize):
            arr = np.array([self.inventory[0], self.backlog[0], self.demand_list[self.step_num - 1]] + self.order[0])/(ACTION_DIM-1)
        else:
            arr = np.array([self.inventory[0], self.backlog[0], self.demand_list[self.step_num - 1]] + self.order[0])
        arr = np.reshape(arr, (self.obs_dim,))
        state.append(arr)

        for i in range(1, self.level_num):
            if(self.normalize):
                arr = np.array([self.inventory[i], self.backlog[i], action[i-1]] + self.order[i])/(ACTION_DIM-1)
            else:
                arr = np.array([self.inventory[i], self.backlog[i], action[i-1]] + self.order[i])
            arr = np.reshape(arr, (self.obs_dim,))
            state.append(arr)
        
        sub_agent_obs = state
        if(self.train):
            sub_agent_reward = [[ALPHA*i + (1-ALPHA)*np.mean(reward)] for i in reward]
        else:
            sub_agent_reward = [[i] for i in reward]

        if(self.train == False and self.step_num == self.eval_eposide_len-1):
            for k in range(LEVEL_NUM):
                if(np.mean(self.action_history[k]) < 1e-6):
                    self.record_act_sta[k].append(0)
                else:
                    self.record_act_sta[k].append(np.std(self.action_history[k])/np.mean(self.action_history[k]))

        if(self.train == False and self.eval_index == 0 and self.step_num == self.eval_eposide_len-1):
            self.eval_bw_res = []
            for i in range(LEVEL_NUM):
                self.eval_bw_res.append(np.mean(self.record_act_sta[i]))
            self.record_act_sta = [[] for i in range(LEVEL_NUM)]

        if(self.step_num > self.eposide_max_steps):
            sub_agent_done = [True for i in range(self.agent_num)]
        else:
            sub_agent_done = [False for i in range(self.agent_num)]
        sub_agent_info = [[] for i in range(self.level_num)]

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
    
    def get_eval_bw_res(self):
        return self.eval_bw_res
    
    def get_eval_num(self):
        return self.n_eval
    
    def get_demand(self):
        return [self.demand_list[self.step_num-1]]
    
    def get_orders(self):
        return self.current_orders
    
    def get_inventory(self):
        return self.inventory
    
    def state_update(self, action):

        self.action = action
        cur_demmand_1 = [self.demand_list[self.step_num]] + action[:-1]
        cur_demmand = [cur_demmand_1[i] + self.backlog[i] for i in range(self.level_num)]
        
        if(self.r_s_l):
            lost_rate = [1-random.randint(0, LOST_RATE*100)/100 for i in range(self.level_num)]
        else:
            lost_rate = [1 for i in range(self.level_num)]

        self.step_num += 1
        reward_ = []
        for i in range(self.level_num):

            unmet = - self.inventory[i] - int(self.order[i][0]*lost_rate[i]) + cur_demmand[i]
            self.inventory[i] = np.max([-unmet, 0])
            self.backlog[i] = np.max([unmet, 0])

            if(i == self.level_num-1):
                self.order[i].append(action[i])
            else:
                self.order[i].append(np.min([cur_demmand[i+1], self.inventory[i+1] + int(self.order[i+1][0]*lost_rate[i+1])]))
            self.action_history[i].append(action[i])
            self.order[i] = self.order[i][1:]
            
            reward = - self.inventory[i]*H[i] - self.backlog[i]*B[i]
            
            reward_.append(reward)

        return reward_