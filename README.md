# Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management
Official codes for ["Multi-Agent Deep Reinforcement Learning for Multi-Echelon Inventory Management"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4262186)

## Create Environment
The codes are implementable on both Windows and Linux with Python 3.8, you can install all required packages with
``` Bash
pip install -r requirements.txt
```


## How To Run
When your environment is ready, you can start the training by running
``` Bash
python train_env.py
```

If everything goes well, you can see the following console outputs
``` Bash
all config:  Namespace(accept_ratio=0.5, algorithm_name='happo', clip_param=0.2, critic_lr=0.0001, cuda=True, cuda_deterministic=True, data_chunk_length=10, entropy_coef=0.01, env_name='MyEnv', episode_length=200, eval_interval=5, experiment_name='check', gae_lambda=0.95, gain=0.01, gamma=0.95, hidden_size=128, huber_delta=10.0, ifi=0.1, kl_threshold=0.01, layer_N=2, log_interval=1, lr=0.0001, ls_step=10, max_grad_norm=0.5, model_dir=None, n_eval_rollout_threads=1, n_no_improvement_thres=10, n_render_rollout_threads=1, n_rollout_threads=5, n_training_threads=1, n_warmup_evaluations=10, num_agents=3, num_env_steps=3000000.0, num_landmarks=3, num_mini_batch=1, opti_eps=1e-05, ppo_epoch=15, recurrent_N=2, render_episodes=5, running_id=1, save_gifs=False, save_interval=1, scenario_name='Ineventory Management', seed=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], seed_specify=True, share_policy=False, stacked_frames=1, use_ReLU=True, use_centralized_V=True, use_clipped_value_loss=True, use_eval=True, use_feature_normalization=False, use_gae=True, use_huber_loss=True, use_linear_lr_decay=False, use_max_grad_norm=False, use_naive_recurrent_policy=True, use_obs_instead_of_state=False, use_orthogonal=True, use_policy_active_masks=False, use_popart=False, use_proper_time_limits=True, use_recurrent_policy=False, use_render=False, use_single_network=False, use_stacked_frames=False, use_value_active_masks=False, 
use_valuenorm=True, user_name='marl', value_loss_coef=0.5, weight_decay=0)
choose to use gpu...
-------------------------------------------------Training starts for seed: 0---------------------------------------------------
share_observation_space:  [Box([-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf
 -inf -inf -inf -inf -inf -inf -inf], [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
 inf inf inf], (21,), float32), Box([-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf
 -inf -inf -inf -inf -inf -inf -inf], [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
 inf inf inf], (21,), float32), Box([-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf
 -inf -inf -inf -inf -inf -inf -inf], [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
 inf inf inf], (21,), float32)]
observation_space:  [Box([-inf -inf -inf -inf -inf -inf -inf], [inf inf inf inf inf inf inf], (7,), float32), Box([-inf -inf -inf -inf -inf -inf -inf], [inf inf inf inf inf inf inf], (7,), float32), Box([-inf -inf -inf -inf -inf -inf -inf], [inf inf inf inf inf inf inf], (7,), float32)]
action_space:  [Discrete(21), Discrete(21), Discrete(21)]

Eval average reward:  -517.6791666666667  Eval ordering fluctuation measurement (downstream to upstream):  [0.5019614682164792, 0.19034909290369945, 0.10862411803628882]

 Algo happo Exp check updates 0/3000 episodes, total num timesteps 1000/3000000.0, FPS 28.

Reward for thread 1: [-215.07, -76.78, -83.41] -125.09  Inventory: [8.0, 48.0, 31.0]  Order: [9.51, 9.7, 9.77] Demand: [9.615]       
Reward for thread 2: [-82.43, -112.2, -114.2] -102.94  Inventory: [69.0, 0.0, 232.0]  Order: [10.54, 9.49, 10.44] Demand: [9.335]    
Reward for thread 3: [-136.65, -66.14, -102.24] -101.68  Inventory: [0.0, 0.0, 161.0]  Order: [9.85, 9.42, 10.11] Demand: [12.46]    
Reward for thread 4: [-496.67, -140.94, -154.17] -263.93  Inventory: [1605.0, 0.0, 119.0]  Order: [9.82, 9.62, 10.17] Demand: [1.725]
Reward for thread 5: [-329.99, -107.07, -100.54] -179.2  Inventory: [0.0, 0.0, 22.0]  Order: [10.42, 9.7, 9.84] Demand: [13.555]
```

The models and results will be saved in a "results" fold outside the project folder

You can change exogenous parameters of the algorithm at [config.py](https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/blob/main/config.py)

## How To Customize Your Own Inventory Management Problem
We provide a template to customize your own problem at [template.py](https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/blob/main/envs/template.py)

We also provide two ready-made supply chain environments, i.e., serial supply chain and supply chain network discussed in our paper, at [serial.py](https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/blob/main/envs/serial.py) and [net_2x3.py](https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/blob/main/envs/net_2x3.py), respectively

Create your environment by following the template and load your environment by importing your environment file at [env_wrappers.py](https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/blob/main/envs/env_wrappers.py) as
``` Bash
#from envs.serial import Env
#from envs.net_2x3 import Env
from envs.your_env import Env
```

*Note. Make sure the "--num_agents" argument at [config.py](https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/blob/main/config.py) is consistent with the number of actors in your environment. For example, if you use the serial supply chain environment [serial.py](https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/blob/main/envs/serial.py), the default value for the "--num_agents" argument at [config.py](https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/blob/main/config.py) should be set to 3; if you use the supply chain network environment [net_2x3.py](https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/blob/main/envs/net_2x3.py), the default value for the "--num_agents" argument at [config.py](https://github.com/xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management/blob/main/config.py) should be set to 6.
