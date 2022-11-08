import numpy as np
from . import generator
import os 
import random

#====================================================================================
# Define all the exogenous parameters you need in your supply chain environment here.
# They may include:
# 1. Number of actors
# 2. Dimention of observation space and action space
# 3. Cost coefficients
# 4. File path of your evaluation demand data
# 5. Other parameters
H = [1, 1, 1, 1, 1]
B = [1, 1, 1, 1, 1]
S_I = 10
S_O = 10
LEAD_TIME = 4
LEVEL_NUM = 3
ACTION_DIM = 21
OBS_DIM = LEAD_TIME + 3
EPISODE_LEN = 200
ALPHA = 0.5

MERTON_DEMAND = True
if(MERTON_DEMAND):
    EVAL_PTH = "./test_data/test_demand_merton/"
else:
    EVAL_PTH = "./test_data/test_demand_stationary/"

RANDOM_SHIPPING_LOSS = False
LOST_RATE = 0.1

PRICE_DISCOUNT = False
DISCOUNT = [1, 0.95, 0.9, 0.85, 0.8]
C = [1, 1, 1, 1, 1]

FIXED_COST = False
F_C = 5
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

def get_training_data():
    """
    - Need to be implemented
    - Load one-episode simulated or local demand data for training
    - Inputs:
        - Modify the inputs as you need
    - Outputs:
        - demand_list: list, one-episode demand data for training
    """
    demand_list = generator.merton(EPISODE_LEN, ACTION_DIM-1)

    return demand_list

class Env(object):

    def __init__(self):

        #============================================================================
        # Define the member variables you need here.
        # The following three memeber variables must be defined
        self.agent_num = LEVEL_NUM
        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM 

        self.inventory = []
        self.backlog = []
        self.order = []
        self.record_act_sta = [[] for i in range(LEVEL_NUM)]
        self.eval_episode_len = EPISODE_LEN
        self.episode_max_steps = EPISODE_LEN
        #============================================================================ 

        self.n_eval, self.eval_data = get_eval_data() # Get demand data for evaluation
        self.eval_index = 0 # Counter for evaluation

    def reset(self, train = True, normalize = True):

        #============================================================================
        # Reset all the member variables that need to be reset at the beginning of an episode here
        # Note that self.eval_index should not be reset here
        # The following one must be reset
        self.step_num = 0

        self.backlog = [0 for i in range(LEVEL_NUM)] 
        self.action_history = [[] for i in range(LEVEL_NUM)]
        self.train = train
        self.normalize = normalize
        self.level_num = LEVEL_NUM
        self.inventory = [S_I for i in range(LEVEL_NUM)]
        self.order = [[S_O for i in range(LEAD_TIME)] for j in range(LEVEL_NUM)]
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

        if(self.step_num > self.episode_max_steps):
            sub_agent_done = [True for i in range(self.agent_num)]
        else:
            sub_agent_done = [False for i in range(self.agent_num)]
        sub_agent_info = [[] for i in range(self.agent_num)]

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
        return [self.demand_list[self.step_num-1]]
    
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

        return self.inventory
    
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
        mapped_actions = []

        self.current_orders = action
        mapped_actions = [i for i in action]

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
            if(normalize):
                arr = np.array([self.inventory[i], 0, S_O] + self.order[i])/(ACTION_DIM-1)
            else:
                arr = np.array([self.inventory[i], 0, S_O] + self.order[i])
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

        if(self.normalize):
            arr = np.array([self.inventory[0], self.backlog[0], self.demand_list[self.step_num - 1]] + self.order[0])/(ACTION_DIM-1)
        else:
            arr = np.array([self.inventory[0], self.backlog[0], self.demand_list[self.step_num - 1]] + self.order[0])
        arr = np.reshape(arr, (self.obs_dim,))
        sub_agent_obs.append(arr)

        for i in range(1, self.level_num):
            if(self.normalize):
                arr = np.array([self.inventory[i], self.backlog[i], action[i-1]] + self.order[i])/(ACTION_DIM-1)
            else:
                arr = np.array([self.inventory[i], self.backlog[i], action[i-1]] + self.order[i])
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
            processed_rewards = [[ALPHA*i + (1-ALPHA)*np.mean(reward)] for i in reward]
        else:
            processed_rewards = [[i] for i in reward]

        return processed_rewards
    
    def state_update(self, action):
        """
        - Need to be implemented
        - Update system state and record some states that you may need in other fuctions like get_eval_bw_res, get_orders, etc.
        - Inputs:
            - action: list, processed actions for each actor
            - Modify the inputs as you need
        - Outputs:
            - rewards: list, rewards for each actors (typically one-period costs for all actors)
        """

        cur_demmand_1 = [self.demand_list[self.step_num]] + action[:-1]
        cur_demmand = [cur_demmand_1[i] + self.backlog[i] for i in range(self.level_num)]
        
        if(RANDOM_SHIPPING_LOSS):
            lost_rate = [1-random.randint(0, LOST_RATE*100)/100 for i in range(self.level_num)]
        else:
            lost_rate = [1 for i in range(self.level_num)]

        self.step_num += 1
        rewards = []
        for i in range(self.level_num):

            unmet = - self.inventory[i] - int(self.order[i][0]*lost_rate[i]) + cur_demmand[i]
            self.inventory[i] = np.max([-unmet, 0])
            self.backlog[i] = np.max([unmet, 0])

            if(i == self.level_num-1):
                self.order[i].append(action[i])
            else:
                self.order[i].append(np.min([cur_demmand[i+1], self.inventory[i+1] + int(self.order[i+1][0]*lost_rate[i+1])]))
            actual_order = self.order[i][-1]
            self.action_history[i].append(action[i])
            self.order[i] = self.order[i][1:]
            
            if(PRICE_DISCOUNT):
                ordering_cost = DISCOUNT[int(action[i]/5)]*C[i]*action[i]
            else:
                ordering_cost = 0

            if(FIXED_COST and actual_order>0):
                fixed_cost = F_C
            else:
                fixed_cost = 0

            reward = - self.inventory[i]*H[i] - self.backlog[i]*B[i] - ordering_cost - fixed_cost

            rewards.append(reward)
        

        if(self.train == False and self.step_num == self.eval_episode_len-1):
            for k in range(LEVEL_NUM):
                if(np.mean(self.action_history[k]) < 1e-6):
                    self.record_act_sta[k].append(0)
                else:
                    self.record_act_sta[k].append(np.std(self.action_history[k])/np.mean(self.action_history[k]))

        if(self.train == False and self.eval_index == 0 and self.step_num == self.eval_episode_len-1):
            self.eval_bw_res = []
            for i in range(LEVEL_NUM):
                self.eval_bw_res.append(np.mean(self.record_act_sta[i]))
            self.record_act_sta = [[] for i in range(LEVEL_NUM)]

        return rewards