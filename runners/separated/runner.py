import time
import numpy as np
from functools import reduce
import torch
from itertools import chain
from runners.separated.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class CRunner(Runner):
    """Runner class to perform training, evaluation. See parent class for details."""
    def __init__(self, config):
        super(CRunner, self).__init__(config)

    def run(self):

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        rewards_log = []
        inv_log = []
        actions_log = []
        demand_log = []
        overall_reward= []
        best_reward = float('-inf')
        best_bw = []
        record = 0

        for episode in range(episodes):
            
            if episode % self.eval_interval == 0 and self.use_eval:
                re, bw_res = self.eval()
                print()
                print("Eval average reward: ", re, " Eval ordering fluctuation measurement (downstream to upstream): ", bw_res)
                if(re > best_reward and episode > 0):
                    self.save()
                    print("A better model is saved!")
                    best_reward = re
                    best_bw = bw_res
                    record = 0
                elif(episode > self.n_warmup_evaluations):
                    record += 1
                    if(record == self.n_no_improvement_thres):
                        print("Training finished because of no imporvement for " + str(self.n_no_improvement_thres) + " evaluations")
                        return best_reward, best_bw

            self.warmup()
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                
                share_obs = []
                for o in obs:
                    share_obs.append(list(chain(*o)))
                
                available_actions = np.array([[None for agent_id in range(self.num_agents)] for info in infos])

                rewards_log.append(rewards)

                inv, demand, orders = self.envs.get_property()
                inv_log.append(inv)
                demand_log.append(demand)
                actions_log.append(orders)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                threads_rew = [[] for i in range(self.n_rollout_threads)]
                threads_inv = [[] for i in range(self.n_rollout_threads)]
                threads_act = [[] for i in range(self.n_rollout_threads)]
                threads_demand = [[] for i in range(self.n_rollout_threads)]
                for i in range(len(rewards_log)):
                    for j in range(self.n_rollout_threads):
                        threads_rew[j].append(rewards_log[i][j])
                        threads_inv[j].append(inv_log[i][j])
                        threads_act[j].append(actions_log[i][j])
                        threads_demand[j].append(demand_log[i][j])
                
                overall_reward.append(np.mean(threads_rew))
                if(len(overall_reward)<6):
                    smooth_reward = overall_reward
                else:
                    smooth_reward = []
                    for i in range(len(overall_reward)-5):
                        smooth_reward.append(np.mean(overall_reward[i:i+10]))
                
                for t in range(len(threads_rew)):
                    rew = [[] for i in range(self.num_agents)]
                    inv = [[] for i in range(self.num_agents)]
                    act = [[] for i in range(self.num_agents)]
                    for i in range(len(threads_rew[t])):
                        for j in range(self.num_agents):
                            rew[j].append(threads_rew[t][i][j])
                            inv[j].append(threads_inv[t][i][j])
                            act[j].append(threads_act[t][i][j])
                    rew = [round(np.mean(l), 2) for l in rew]
                    inv = [round(np.mean(l), 2) for l in inv]
                    act = [round(np.mean(l), 2) for l in act]
                    print("Reward for thread " + str(t+1) + ": " + str(rew) + " " + str(round(np.mean(rew),2))+"  Inventory: " + str(inv)+"  Order: " + str(act) + " Demand: " + str(np.mean(threads_demand[t], 0)))
                rewards_log = []
                inv_log = []
                actions_log = []
                demand_log = []
            # eval

    def warmup(self):
        # reset env
        obs, available_actions = self.envs.reset()
        # replay buffer
        
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents): 
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()
            self.buffer[agent_id].available_actions[0] = None

    @torch.no_grad()
    def collect(self, step):
        value_collector=[]
        action_collector=[]
        temp_actions_env = []
        action_log_prob_collector=[]
        rnn_state_collector=[]
        rnn_state_critic_collector=[]

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                self.buffer[agent_id].obs[step],
                                                self.buffer[agent_id].rnn_states[step],
                                                self.buffer[agent_id].rnn_states_critic[step],
                                                self.buffer[agent_id].masks[step],
                                                self.buffer[agent_id].available_actions[step])

            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))

            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action.cpu().detach()], 1)
            else:
                raise NotImplementedError
            
            temp_actions_env.append(action_env)

            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
        
        # [self.envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
                
            self.buffer[agent_id].insert(share_obs, np.array(list(obs[:, agent_id])), rnn_states[:,agent_id],
                    rnn_states_critic[:,agent_id],actions[:,agent_id], action_log_probs[:,agent_id],
                    values[:,agent_id], rewards[:,agent_id], masks[:,agent_id])

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self):
        
        overall_reward = []
        eval_num = self.eval_envs.get_eval_num()

        for _ in range(eval_num):
            eval_obs, eval_available_actions = self.eval_envs.reset()
            
            eval_share_obs = []
            for o in eval_obs:
                eval_share_obs.append(list(chain(*o)))
            eval_share_obs = np.array(eval_share_obs)

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for eval_step in range(self.episode_length):
                temp_actions_env = []

                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    eval_actions, temp_rnn_state = \
                        self.trainer[agent_id].policy.act(eval_obs[:,agent_id],
                                                eval_rnn_states[:,agent_id],
                                                eval_masks[:,agent_id],
                                                None,
                                                deterministic=True)
                    eval_rnn_states[:,agent_id]=_t2n(temp_rnn_state)
                    action = eval_actions.detach().cpu().numpy()

                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)

                #eval_actions = np.array(eval_actions_collector).transpose(1,0,2)
                eval_actions_env = []
                for i in range(self.n_eval_rollout_threads):
                    eval_one_hot_action_env = []
                    for eval_temp_action_env in temp_actions_env:
                        eval_one_hot_action_env.append(eval_temp_action_env[i])
                    eval_actions_env.append(eval_one_hot_action_env)

                # Obser reward and next obs
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
                
                eval_share_obs = []
                for o in eval_obs:
                    eval_share_obs.append(list(chain(*o)))
                eval_share_obs = np.array(eval_share_obs)

                eval_available_actions = None

                overall_reward.append(np.mean(eval_rewards))

                eval_dones_env = np.all(eval_dones, axis=1)

                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        
        bw_res = self.eval_envs.get_eval_bw_res()
        return np.mean(overall_reward), bw_res