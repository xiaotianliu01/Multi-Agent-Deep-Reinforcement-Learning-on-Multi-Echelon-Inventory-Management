import numpy as np
from . import generator
import os 
import math

#====================================================================================
# Define all the exogenous parameters you need in your supply chain environment here.
# They may include:
# 1. Number of actors
# 2. Dimention of observation space and action space
# 3. Cost coefficients
# 4. File path of your evaluation demand data
# 5. Other parameters

C = [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]
H = [1, 1, 1, 1, 1, 1]
B = [1, 1, 1, 1, 1, 1]

S_I = 2
S_O = 2
LEAD_TIME = 4
LEVEL_NUM = 3
ACTION_DIM = 25
OBS_DIM = LEAD_TIME + 4
ALPHA = 0.75
EVAL_PTH = ["./test_data/test_demand_net/0/", "./test_data/test_demand_net/1/"]
EPOSIDE_LEN = 200

#====================================================================================

def get_eval_data():
    """
    - Need to be implemented
    - Load local demand data for evaluation
    - Inputs:
        - Modify the inputs as you need
    - Outputs:
        - n_eval: int, number of demand sequences (also number of episodes in one evaluation)
        - eval_data: list, demand data for evaluation
    """

    files_0 = os.listdir(EVAL_PTH[0])
    files_1 = os.listdir(EVAL_PTH[1])
    n_eval = len(files_0)
    eval_data = []

    for i in range(n_eval):
        data_0 = []
        with open(EVAL_PTH[0] + files_0[i], "rb") as f:
            lines = f.readlines()
            for line in lines:
                data_0.append(int(line))
        
        data_1 = []
        with open(EVAL_PTH[1] + files_1[i], "rb") as f:
            lines = f.readlines()
            for line in lines:
                data_1.append(int(line))
        eval_data.append([data_0, data_1])

    return n_eval, eval_data

def get_training_data():
    """
    - Need to be implemented
    - Load one-episode simulated or local demand data for training
    - Inputs:
        - Modify the inputs as you need
    - Outputs:
        - demand_list: list, one-episode demand data for training
    """
    demand_list = [generator.merton(EPOSIDE_LEN, math.sqrt(ACTION_DIM)-1), generator.merton(EPOSIDE_LEN, math.sqrt(ACTION_DIM)-1)]

    return demand_list

class Env(object):

    def __init__(self):

        #============================================================================
        # Define the member variables you need here.
        # The following three memeber variables must be defined
        self.agent_num = LEVEL_NUM*2  
        self.obs_dim = OBS_DIM 
        self.action_dim = ACTION_DIM 

        self.inventory = []
        self.order = []
        self.record_act_sta = [[] for i in range(2*LEVEL_NUM)]
        self.eposide_max_steps = EPOSIDE_LEN
        self.eval_eposide_len = EPOSIDE_LEN
        #============================================================================ 

        self.n_eval, self.eval_data  = get_eval_data()
        self.eval_index = 0
        
    def reset(self, train = True, normalize = True):

        #============================================================================
        # Reset all the member variables that need to be reset at the beginning of an episode here
        # Note that self.eval_index should not be reset here
        # The following one must be reset
        self.step_num = 0

        self.backlog = [[0, 0] for i in range(LEVEL_NUM*2)]
        self.action_history = [[[],[]] for i in range(LEVEL_NUM*2)]

        self.train = train
        self.normalize = normalize
    
        self.level_num = LEVEL_NUM
        self.inventory = [[S_I, S_I] for i in range(LEVEL_NUM)]
        self.order = [[[S_O for i in range(LEAD_TIME)], [S_O for i in range(LEAD_TIME)]] for j in range(LEVEL_NUM)]
        #============================================================================

        if(train == True):
            self.demand_list = get_training_data() # Get demand data for training
        else:
            self.demand_list = self.eval_data[self.eval_index] # Get demand data for evaluation

            self.eval_index += 1
            if(self.eval_index == self.n_eval):
                self.eval_index = 0

        sub_agent_obs = self.get_reset_obs(normalize) # Get reset obs
            
        return sub_agent_obs

    def step(self, actions, one_hot = True):

        if(one_hot):
            action_ = [np.argmax(i) for i in actions]
        else:
            action_ = actions

        action = self.action_map(action_) # Map outputs of MADRL to actual ordering actions
        reward = self.state_update(action) # System state update
        sub_agent_obs = self.get_step_obs(action) # Get step obs
        sub_agent_reward = self.get_processed_rewards(reward) # Get processed rewards
        
        if(self.step_num > self.eposide_max_steps):
            sub_agent_done = [True for i in range(self.agent_num)]
        else:
            sub_agent_done = [False for i in range(self.agent_num)]
        sub_agent_info = [[] for i in range(self.level_num)]

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
    
    def get_eval_num(self):
        return self.n_eval

    def get_eval_bw_res(self):
        """"
        - Need to be implemented
        - Get the ordering fluctuation measurement for each actor/echelon during evaluation. The results will be printed out after each evaluation during training. 
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - eval_bw_res: list, ordering fluctuation measurement for each actor/echelon
        """
        return self.eval_bw_res
    
    def get_demand(self):
        return [self.demand_list[0][self.step_num-1], self.demand_list[1][self.step_num-1]]
    
    def get_orders(self):
        """"
        - Need to be implemented
        - Get actual ordering actions for all actors. Will be printed out during the training.
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - current_orders: list, actual ordering actions for all actors
        """
        return self.current_orders
    
    def get_inventory(self):
        """"
        - Need to be implemented
        - Get inventory levels for all actors. Will be printed out during the training.
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - current_inventory: list, inventory levels for all actors
        """
        inv = []
        for i in range(self.level_num):
            inv.append(self.inventory[i][0])
            inv.append(self.inventory[i][1])
        return inv
    
    def action_map(self, action):
        """
        - Need to be implemented
        - Map the output of MADRL to actucal ordering actions 
        - Inputs:
            - action: list, output of MADRL
            - Modify the inputs as you need
        - Outputs:
            - mapped_actions: list, actual ordering actions
        """
        mapped_actions = [[int(i/math.sqrt(ACTION_DIM)), int(i%math.sqrt(ACTION_DIM))] for i in action]
        self.current_orders = [np.sum(act) for act in mapped_actions]
        for i in range(LEVEL_NUM):
            for j in range(2):
                self.action_history[i*2+j][0].append(mapped_actions[i*2+j][0])
                self.action_history[i*2+j][1].append(mapped_actions[i*2+j][1])

        return mapped_actions
    
    def get_reset_obs(self, normalize):
        """
        - Need to be implemented
        - Get reset obs (initial obs)
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - sub_agent_obs: list, a list for obs of all actors, shape for obs of each actor: (self.obs_dim, )
        """
        sub_agent_obs = []
        for i in range(self.level_num):
            for j in range(2):
                if(normalize):
                    arr = np.array([self.inventory[i][j],0,0,S_O] + self.order[i][j])/(math.sqrt(ACTION_DIM)-1)
                else:
                    arr = np.array([self.inventory[i][j],0,0,S_O] + self.order[i][j])
                arr = np.reshape(arr, (self.obs_dim,))
                sub_agent_obs.append(arr)

        return sub_agent_obs
    
    def get_step_obs(self, action):
        """
        - Need to be implemented
        - Get step obs (obs for each step)
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - sub_agent_obs: list, a list for obs of all actors, shape for obs of each actor: (self.obs_dim, )
        """
        sub_agent_obs = []

        for i in range(0, self.level_num-1):
            for j in range(2):
                if(self.normalize):
                    arr = np.array([self.inventory[i][j], self.backlog[i*2+j][0] + self.backlog[i*2+j][1], self.backlog[(i+1)*2][j]+self.backlog[(i+1)*2+1][j], action[(i-1)*2][j]+action[(i-1)*2+1][j]] + self.order[i][j])/(math.sqrt(ACTION_DIM)-1)
                else:
                    arr = np.array([self.inventory[i][j], self.backlog[i*2+j][0] + self.backlog[i*2+j][1], self.backlog[(i+1)*2][j]+self.backlog[(i+1)*2+1][j], action[(i-1)*2][j]+action[(i-1)*2+1][j]] + self.order[i][j])
                arr = np.reshape(arr, (self.obs_dim,))
                sub_agent_obs.append(arr)
        
        i = self.level_num-1
        
        for j in range(2):
                if(self.normalize):
                    arr = np.array([self.inventory[i][j], self.backlog[i*2+j][0] + self.backlog[i*2+j][1], 0, action[(i-1)*2][j]+action[(i-1)*2+1][j]] + self.order[i][j])/(math.sqrt(ACTION_DIM)-1)
                else:
                    arr = np.array([self.inventory[i][j], self.backlog[i*2+j][0] + self.backlrecord_staog[i*2+j][1], 0, action[(i-1)*2][j]+action[(i-1)*2+1][j]] + self.order[i][j])
                arr = np.reshape(arr, (self.obs_dim,))
                sub_agent_obs.append(arr)

        return sub_agent_obs
    
    def get_processed_rewards(self, reward):
        """
        - Need to be implemented
        - Get processed rewards for all actors
        - Inputs:
            - reward: list, reward directly from the state update (typically each actor's on-period cost)
            - Modify the inputs as you need
        - Outputs:
            - processed_rewards: list, a list for rewards of all actors
        """
        processed_rewards = []
        if(self.train):
            processed_rewards = [[ALPHA*i+(1-ALPHA)*np.mean(reward)] for i in reward]
            
        else:
            processed_rewards = [[i] for i in reward]

        return processed_rewards

    def state_update(self, action_):
        """
        - Need to be implemented
        - Update system state and record some states that you may need in other fuctions like get_eval_bw_res, get_orders, etc.
        - Inputs:
            - action: list, processed actions for each actor
            - Modify the inputs as you need
        - Outputs:
            - rewards: list, rewards for each actors (typically one-period costs for all actors)
        """
        
        action = []
        for i in range(self.level_num):
            action.append([action_[i*2], action_[i*2+1]])
        cur_demmand = [[[self.demand_list[0][self.step_num], 0], [0, self.demand_list[1][self.step_num]]]]

        for i in range(1, self.level_num):
            de = []
            de.append([action_[(i-1)*2][0] + self.backlog[(i)*2][0], action_[(i-1)*2+1][0]+self.backlog[(i)*2][1]])
            de.append([action_[(i-1)*2][1] + self.backlog[(i)*2+1][0], action_[(i-1)*2+1][1]+self.backlog[(i)*2+1][1]])
            cur_demmand.append(de)

        self.step_num += 1
        rewards = [0,0,0,0,0,0]
        sale_s2c = action[self.level_num-1]
        for i in range(self.level_num-1, -1, -1):
            t_sale_s2c = [[0,0],[0,0]]

            for j in range(2):

                E_S = np.min([self.inventory[i][j] + self.order[i][j][0], cur_demmand[i][j][1]])
                N_S = np.min([cur_demmand[i][j][0], self.inventory[i][j] + self.order[i][j][0] - E_S])

                t_sale_s2c[0][j] = N_S
                t_sale_s2c[1][j] = E_S

                self.backlog[i*2+j][0] = cur_demmand[i][j][0] - N_S
                self.backlog[i*2+j][1] = cur_demmand[i][j][1] - E_S

                lost_sales = -self.inventory[i][j] - self.order[i][j][0] + np.sum(cur_demmand[i][j])
                self.inventory[i][j] = np.max([-lost_sales, 0])

                self.order[i][j].append(sale_s2c[j][0])
                self.order[i][j] = self.order[i][j][1:]
                self.order[i][j][1] += sale_s2c[j][1]
                actual_order = [sale_s2c[j][0], sale_s2c[j][1]]

                b_c = B[i*2+j]*np.sum(self.backlog[(i)*2+j])
                reward = - actual_order[0]*C[i][0] - actual_order[1]*C[i][1] - self.inventory[i][j]*H[i] - b_c
                rewards[i*2+j] = reward

            sale_s2c = t_sale_s2c
        
        if(self.train == False and self.step_num == self.eval_eposide_len-1):
            for k in range(LEVEL_NUM):
                for j in range(2):
                    if(np.mean(self.action_history[k*2+j][0]) + np.mean(self.action_history[k*2+j][1]) < 1e-6):
                        self.record_act_sta[k*2+j].append(0)
                    else:
                        tem = [self.action_history[k*2+j][0][i] + self.action_history[k*2+j][1][i] for i in range(len(self.action_history[k*2+j][1]))]
                        self.record_act_sta[k*2+j].append(np.std(tem)/np.mean(tem))

        if(self.train == False and self.eval_index == 0 and self.step_num == self.eval_eposide_len-1):
            self.eval_bw_res = []
            for i in range(LEVEL_NUM):
                temp = []
                for j in range(2):
                    temp.append(np.mean(self.record_act_sta[i*2+j]))
                self.eval_bw_res.append(np.mean(temp))
            self.record_act_sta = [[] for i in range(2*LEVEL_NUM)]
        
        return rewards