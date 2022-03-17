from unstable_baselines.common.util import second_to_time_str
from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import trange
from time import time
import cv2
import os
import random
import torch
from unstable_baselines.common import util 

class PEARLTrainer(BaseTrainer):
    def __init__(self, agent, env, train_task_indices, eval_task_indices, train_replay_buffers, train_encoder_buffers, eval_buffer, load_dir,
            batch_size,
            z_inference_batch_size,
            use_next_obs_in_context,
            max_epoch,
            num_tasks_per_gradient_update,
            num_train_tasks,
            num_eval_tasks,
            num_train_tasks_to_sample_per_iteration, 
            num_steps_prior, 
            num_steps_posterior, 
            num_extra_rl_steps_posterior,
            num_updates_per_iteration,
            adaptation_context_update_interval,
            num_eval_posterior_trajs,
            start_timestep,
            num_eval_rounds,
            **kwargs):
        super(PEARLTrainer, self).__init__(agent, env, env, **kwargs) #train env is the same as the eval env, train and eval tasks are identified by their indicies
        self.agent = agent
        self.train_replay_buffers = train_replay_buffers
        self.train_encoder_buffers = train_encoder_buffers
        self.eval_buffer = eval_buffer
        self.env = env 
        self.train_task_indices = train_task_indices
        self.eval_task_indices = eval_task_indices
        #hyperparameters
        self.batch_size = batch_size
        self.z_inference_batch_size = z_inference_batch_size
        self.use_next_obs_in_context = use_next_obs_in_context
        self.num_train_tasks = num_train_tasks
        self.num_eval_tasks = num_eval_tasks
        self.max_epoch = max_epoch
        self.num_tasks_per_gradient_update = num_tasks_per_gradient_update
        self.num_train_tasks_to_sample_per_iteration = num_train_tasks_to_sample_per_iteration
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_updates_per_iteration = num_updates_per_iteration
        self.adaptation_context_update_interval = adaptation_context_update_interval
        self.num_eval_posterior_trajs = num_eval_posterior_trajs
        self.num_eval_rounds = num_eval_rounds
        self.start_timestep = start_timestep
        if load_dir != "":
            if  os.path.exists(load_dir):
                self.agent.load(load_dir)
            else:
                print("Load dir {} Not Found".format(load_dir))
                exit(0)

    def warmup(self):
        for task_idx in self.train_task_indices:
            self.env.reset_task(task_idx)
            initial_z = {
                'z_mean': torch.zeros((1, self.agent.latent_dim), device=util.device),
                'z_var': torch.ones((1, self.agent.latent_dim), device=util.device)
                }
            self.collect_data(task_idx, num_samples=self.start_timestep, resample_z_rate=1, update_posterior_rate=np.inf, is_training=True, initial_z=initial_z, add_to_enc_buffer=True)

    def train(self):
        train_traj_returns = [0]
        #collect initial pool of data
        self.warmup()
        tot_env_steps = self.start_timestep * self.num_train_tasks
        for epoch_id in trange(self.max_epoch, colour='blue', desc='outer loop'): 
            self.pre_iter()
            log_infos = {}

            train_task_returns = []
            #sample data from train_tasks
            for idx in range(self.num_train_tasks_to_sample_per_iteration):
                train_task_idx = random.choice(self.train_task_indices) # the same sampling method as the source code
                self.env.reset_task(train_task_idx)
                self.train_encoder_buffers[train_task_idx].clear()
                prior_samples, posterior_samples, extra_posterior_samples = {"obs_list":[]} ,{"obs_list":[]}, {"obs_list":[]}

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    initial_z = {
                        'z_mean': torch.zeros((1, self.agent.latent_dim), device=util.device),
                        'z_var': torch.ones((1, self.agent.latent_dim), device=util.device)
                        }
                    self.collect_data(train_task_idx, self.num_steps_prior, resample_z_rate=1, update_posterior_rate=np.inf, is_training=True, initial_z=initial_z, add_to_enc_buffer=True)

                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(train_task_idx, self.num_steps_posterior, resample_z_rate=1, update_posterior_rate=self.adaptation_context_update_interval, is_training=True, add_to_enc_buffer=True)

                # trajectories from the policy handling z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    extra_posterior_returns = self.collect_data(train_task_idx, self.num_extra_rl_steps_posterior, resample_z_rate=1, update_posterior_rate=self.adaptation_context_update_interval, is_training=True, add_to_enc_buffer=False)
                    #does not add to encoder buffer since it's extra posterior
                
                tot_env_steps += self.num_steps_prior + self.num_steps_posterior + self.num_extra_rl_steps_posterior
                train_task_returns.append(np.mean(extra_posterior_returns))
            # info_str = ""
            # for i in range(self.num_train_tasks):
            #     info_str += "{}: {}/{}\t".format(i, self.train_replay_buffers[i].max_sample_size, self.train_encoder_buffers[i].max_sample_size)
            #     if (i+1) % 10 == 0:
            #         info_str += "\n"
            # print(info_str)
            #perform training on batches of train tasks
            for train_step in trange(self.num_updates_per_iteration, colour='green', desc='inner loop'):
                train_task_indices = np.random.choice(range(self.num_train_tasks), self.num_tasks_per_gradient_update)
                train_agent_loss_dict = self.train_step(train_task_indices)

            train_traj_returns.append(np.mean(train_task_returns))
            log_infos['performance/train_return'] = train_traj_returns[-1]
            log_infos.update(train_agent_loss_dict)
            self.post_iter(log_infos, tot_env_steps)

    def train_step(self, train_task_indices):
        # sample train context
        context_batch = self.sample_context(train_task_indices, is_training=True)
        
        #sample data for sac update
        data_batch = [self.train_replay_buffers[idx].sample(batch_size=self.batch_size) for idx in train_task_indices]
        data_batch = [self.unpack_context_batch(data) for data in data_batch]
        data_batch = [[x[i] for x in data_batch] for i in range(len(data_batch[0]))]
        data_batch = [torch.cat(x, dim=0) for x in data_batch]
        # obs, actions, next_obs, reward, done
        
        #perform update on context and data, and also passes task indices for log
        loss_dict = self.agent.update(train_task_indices, context_batch, data_batch)
        
        return loss_dict

    def sample_context(self, indices, is_training):
        if is_training:
            contexts = [self.train_encoder_buffers[idx].sample(batch_size=self.z_inference_batch_size) for idx in indices]
            # task_num,  [states, actions, next_states, rewards, dones] of length batch_size
        else:
            contexts = [self.eval_buffer.sample(self.z_inference_batch_size)]
        contexts = [self.unpack_context_batch(context_batch) for context_batch in contexts]
        contexts = [[x[i] for x in contexts] for i in range(len(contexts[0]))]
        contexts = [torch.cat(x, dim=0) for x in contexts]
        if self.use_next_obs_in_context:
            contexts = torch.cat(contexts[:-1], dim=2)
        else:
            contexts = torch.cat(contexts[:-2], dim=2)
        # if not is_training:
        #     print("contexts:", contexts.shape)
        return contexts
    
    def unpack_context_batch(self, data):
        ''' unpack a batch and return individual elements '''
        o = data['obs'][None, ...]
        a = data['action'][None, ...]
        r = data['reward'][None, ...]
        no = data['next_obs'][None, ...]
        d = data['done'][None, ...]
        return [o, a, r, no, d]

    def calculate_data_return(self, data):
        reward_list = data['reward_list']
        done_list = data['done_list']
        return_list = []
        curr_traj_return = 0
        for r,d in zip(reward_list, done_list):
            curr_traj_return += r
            if d:
                return_list.append(curr_traj_return)
                curr_traj_return = 0
        return return_list
                
    def evaluate(self):
        #copy env parameters from train env to eval env
        task_traj_returns = []
        for idx in self.eval_task_indices:
            traj_returns = []
            for round in range(self.num_eval_rounds):
                self.eval_env.reset_task(idx)
                self.eval_buffer.clear()
                
                # prior sample
                initial_z = {
                    "z_mean": torch.zeros((1, self.agent.latent_dim), device=util.device), 
                    "z_var": torch.ones((1, self.agent.latent_dim), device=util.device)
                }
                self.collect_data(idx,num_samples=self.max_trajectory_length, 
                                                            resample_z_rate=1,  update_posterior_rate=np.inf, 
                                                            is_training=False, initial_z=initial_z)

                # posterior sample
                for _ in range(self.num_eval_posterior_trajs): 
                    self.collect_data(idx, num_samples=self.max_trajectory_length, resample_z_rate=1, update_posterior_rate=self.adaptation_context_update_interval,is_training=False)

                # eval
                context = self.sample_context(None, is_training=False)
                z_means, z_vars = self.agent.infer_z_posterior(context)
                z = self.agent.sample_z_from_posterior(z_means, z_vars)
                for traj_idx in range(self.num_eval_trajectories):
                    eval_data, traj_return = self.rollout_trajectory(z, deterministic=True)
                    traj_returns.append(traj_return)
            
            task_traj_returns.append(np.mean(traj_returns))

        eval_traj_mean_return = np.mean(task_traj_returns)
        return {
            "performance/eval_return": eval_traj_mean_return
        }                                                          

    def collect_data(self, task_idx, num_samples, resample_z_rate, update_posterior_rate, is_training, initial_z=None, add_to_enc_buffer=False):
        num_samples_collected = 0
        num_trajectories_collected = 0
        if initial_z is None:
            z_inference_context_batch = self.sample_context([task_idx], is_training=is_training)
            z_mean, z_var = self.agent.infer_z_posterior(z_inference_context_batch)
            z_sample = self.agent.sample_z_from_posterior(z_mean, z_var)
        else: # for initialization when no context is available in the buffer
            z_mean = initial_z['z_mean']
            z_var = initial_z['z_var']
            z_sample = self.agent.sample_z_from_posterior(z_mean, z_var)
        traj_returns = []
        while num_samples_collected < num_samples: 
            deterministic = not is_training
            #rollout one trajectory every time
            curr_traj_samples, traj_return = self.rollout_trajectory(z_sample, deterministic=deterministic)
            num_samples_collected += len(curr_traj_samples['obs_list'])
            traj_returns.append(traj_return)
            num_trajectories_collected += 1
            #add to buffer
            if is_training:
                self.train_replay_buffers[task_idx].add_traj(**curr_traj_samples)
                if add_to_enc_buffer:
                    self.train_encoder_buffers[task_idx].add_traj(**curr_traj_samples)
            else:
                self.eval_buffer.add_traj(**curr_traj_samples)

            if update_posterior_rate !=np.inf and num_trajectories_collected % update_posterior_rate == 0:
                #update z posterior inference
                z_inference_context_batch = self.sample_context([task_idx], is_training=is_training)
                z_mean, z_var = self.agent.infer_z_posterior(z_inference_context_batch)
                z_sample = self.agent.sample_z_from_posterior(z_mean, z_var)
            if num_trajectories_collected % resample_z_rate == 0:
                z_sample = self.agent.sample_z_from_posterior(z_mean, z_var)
        return traj_returns

    def rollout_trajectory(self, z, deterministic=False):
        obs_list, action_list, next_obs_list, reward_list, done_list = [], [], [], [], []
        done = False
        obs = self.env.reset()
        traj_length = 0
        traj_return = 0
        while not done and traj_length < self.max_trajectory_length:
            action = self.agent.select_action(obs, z, deterministic=deterministic)['action']
            next_obs, reward, done, info = self.env.step(action)
            traj_return += reward
            obs_list.append(obs)
            action_list.append(action)
            next_obs_list.append(next_obs)
            reward_list.append(reward)
            if traj_length >= self.max_trajectory_length - 1:
                done = False
            done_list.append(done)
            traj_length += 1

            # IMPORTANT: remember to step the observation :)
            obs = next_obs
        return {
            'obs_list': obs_list, 
            'action_list': action_list, 
            'next_obs_list': next_obs_list, 
            'reward_list': reward_list, 
            'done_list': done_list
        }, traj_return
        

    def save_video_demo(self, ite, width=128, height=128, fps=30):
        video_demo_dir = os.path.join(util.logger.log_dir,"demos")
        if not os.path.exists(video_demo_dir):
            os.makedirs(video_demo_dir)
        ite_video_demo_dir = os.path.join(video_demo_dir, "ite_{}".format(ite))
        os.mkdir(ite_video_demo_dir)
        video_size = (height, width)
        for idx in self.eval_task_indices:
            #reset env and buffer
            self.eval_env.reset_task(idx)
            self.eval_buffer.clear()
            initial_z = {
                        'z_mean': torch.zeros((1, self.agent.latent_dim), device=util.device),
                        'z_var': torch.ones((1, self.agent.latent_dim), device=util.device)
                        }
            self.collect_data(idx, self.num_steps_prior, 1, np.inf, is_training=False, initial_z=initial_z)
            self.collect_data(idx, self.num_steps_posterior, 1, self.adaptation_context_update_interval, is_training=False)
            context = self.sample_context(None, is_training=False) 
            z_mean, z_var = self.agent.infer_z_posterior(context)
            z = self.agent.sample_z_from_posterior(z_mean, z_var)
            video_save_path = os.path.join(ite_video_demo_dir, "task_{}.mp4".format(idx))

            #initilialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, video_size)

            #rollout to generate pictures and write video
            obs = self.eval_env.reset()
            img = self.eval_env.render(mode="rgb_array", width=width, height=height)
            for step in range(self.max_trajectory_length):
                action = self.agent.select_action(obs, z, deterministic=True)['action']
                next_obs, reward, done, _ = self.eval_env.step(action)
                obs = next_obs
                img = self.eval_env.render(mode="rgb_array", width=width, height=height)
                video_writer.write(img)
                if done:
                    break 
            video_writer.release()
    
    def visualize_task_z(self):
        pass