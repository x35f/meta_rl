import torch
import torch.nn.functional as F
from torch import nn
from unstable_baselines.common.agents import BaseAgent
from unstable_baselines.common.networks import SequentialNetwork, PolicyNetworkFactory, get_optimizer
from .network import TanhGaussianPolicy
import numpy as np
from unstable_baselines.common import util, functional
from unstable_baselines.common.maths import product_of_gaussians
from operator import itemgetter
from gym.spaces import Box

class PEARLAgent(BaseAgent):
    def __init__(self, observation_space, action_space,
        alpha,
        reward_scale,
        target_smoothing_tau,
        kl_lambda,
        **kwargs):
        super(PEARLAgent, self).__init__()
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        #save parameters
        self.args = kwargs

        #initilze networks
        self.latent_dim = kwargs['latent_dim']
        self.q1_network = SequentialNetwork(obs_dim + action_dim + self.latent_dim, 1, **kwargs['q_network'])
        self.q2_network = SequentialNetwork(obs_dim + action_dim + self.latent_dim, 1,**kwargs['q_network'])
        self.target_q1_network = SequentialNetwork(obs_dim + action_dim + self.latent_dim, 1, **kwargs['q_network'])
        self.target_q2_network = SequentialNetwork(obs_dim + action_dim + self.latent_dim, 1,**kwargs['q_network'])
        policy_input_space = Box(low=observation_space.low[0], high=observation_space.high[0], dtype=observation_space.dtype, shape=(obs_dim + self.latent_dim,))
        self.policy_network = PolicyNetworkFactory.get(policy_input_space, action_space,  ** kwargs['policy_network'])
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']
        if self.use_next_obs_in_context:
            context_encoder_input_dim = 2 * obs_dim + action_dim + 1
        else:
            context_encoder_input_dim =  obs_dim + action_dim + 1
        self.use_information_bottleneck = kwargs['use_information_bottleneck']
        if self.use_information_bottleneck:
            context_encoder_output_dim = kwargs['latent_dim'] * 2
        else:
            context_encoder_output_dim = kwargs['latent_dim']
        self.context_encoder_network = SequentialNetwork(context_encoder_input_dim, context_encoder_output_dim, **kwargs['context_encoder_network'])

        functional.soft_update_network(self.q1_network, self.target_q1_network, 1.0)
        functional.soft_update_network(self.q2_network, self.target_q2_network, 1.0)

        #pass to util.device
        self.q1_network = self.q1_network.to(util.device)
        self.q2_network = self.q2_network.to(util.device)
        self.target_q1_network = self.target_q1_network.to(util.device)
        self.target_q2_network = self.target_q2_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)
        self.context_encoder_network = self.context_encoder_network.to(util.device)
        
        #register networks
        self.networks = {
            'q1_network': self.q1_network,
            'q2_network': self.q2_network,
            'target_q1_network': self.target_q1_network,
            'target_q2_network': self.target_q2_network,
            'policy_network': self.policy_network,
            'context_encoder_network': self.context_encoder_network
        }

        #initialize optimizer
        self.q1_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q1_network, kwargs['q_network']['learning_rate'])
        self.q2_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q2_network, kwargs['q_network']['learning_rate'])
      
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])
        self.context_encoder_optimizer = get_optimizer(kwargs['context_encoder_network']['optimizer_class'], self.context_encoder_network, kwargs['context_encoder_network']['learning_rate'])

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.reward_scale = reward_scale
        self.kl_lambda =kl_lambda
        self.target_smoothing_tau = target_smoothing_tau

        #entropy
        self.automatic_entropy_tuning = kwargs['entropy']['automatic_tuning']
        self.alpha = alpha
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -np.prod(action_space.shape).item()
            self.log_alpha = torch.zeros(1, device=util.device)
            self.log_alpha = nn.Parameter(self.log_alpha, requires_grad=True)
            # sel5f.log_alpha = torch.FloatTensor(math.log(alpha), requires_grad=True, device=util.device)
            self.alpha = self.log_alpha.detach().exp()
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=kwargs['entropy']['learning_rate'])

    def update(self, train_task_indices, context_batch, data_batch):
        
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = data_batch
        
        # infer z from context
        z_means, z_vars = self.infer_z_posterior(context_batch)
        task_z_batch = self.sample_z_from_posterior(z_means, z_vars)
        # expand z to concatenate with obs, action batches
        num_tasks, batch_size, obs_dim = obs_batch.shape

        #flatten obs
        obs_batch = obs_batch.view(num_tasks * batch_size, -1)
        action_batch = action_batch.view(num_tasks * batch_size, -1)
        next_obs_batch = next_obs_batch.view(num_tasks * batch_size, -1)
        reward_batch = reward_batch.view(num_tasks * batch_size, -1) * self.reward_scale
        done_batch = done_batch.view(num_tasks * batch_size, -1)
        
        #expand z to match obs batch
        task_z_batch = torch.stack([task_z.repeat(batch_size, 1) for task_z in task_z_batch]).view(num_tasks * batch_size, -1)
        
        #compute Q loss and context encoder loss
        curr_state_q1_value = self.q1_network(torch.cat([obs_batch, action_batch, task_z_batch], dim=1))
        curr_state_q2_value = self.q2_network(torch.cat([obs_batch, action_batch, task_z_batch], dim=1))
        with torch.no_grad():
            policy_input = torch.cat([next_obs_batch, task_z_batch.detach()], dim=1)
            next_obs_action, next_state_log_pi = \
                itemgetter("action_scaled", "log_prob")(self.policy_network.sample(policy_input))
            next_state_q1_value = self.target_q1_network(torch.cat([next_obs_batch, next_obs_action, task_z_batch], dim=1))
            next_state_q2_value = self.target_q2_network(torch.cat([next_obs_batch, next_obs_action, task_z_batch], dim=1))
            next_state_min_q = torch.min(next_state_q1_value, next_state_q2_value)
            target_q_value = (next_state_min_q - self.alpha * next_state_log_pi)
            target_q_value = reward_batch + self.gamma * (1. - done_batch) * target_q_value

        #compute q loss
        q1_loss = F.mse_loss(curr_state_q1_value, target_q_value.detach())
        q2_loss =  F.mse_loss(curr_state_q2_value, target_q_value.detach())
        q_loss = q1_loss + q2_loss

        #compute kl loss if use information bottleneck for context optimizer
        prior = torch.distributions.Normal(torch.zeros(self.latent_dim, device=util.device), torch.ones(self.latent_dim, device=util.device))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(z_means), torch.unbind(z_vars))] #todo: inpect std
        kl_divs = torch.stack([torch.distributions.kl.kl_divergence(post, prior) for post in posteriors])
        kl_div = torch.sum(kl_divs)
        kl_loss = self.kl_lambda * kl_div

        # update context encoder and q network
        self.context_encoder_optimizer.zero_grad()
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_loss.backward(retain_graph=True)
        q_loss.backward()
        self.context_encoder_optimizer.step()
        self.q1_optimizer.step()
        self.q2_optimizer.step()


        # compute loss w.r.t policy
        policy_input = torch.cat([obs_batch, task_z_batch.detach()], dim=1)
        new_curr_state_action, new_curr_state_log_pi = \
            itemgetter("action_scaled", "log_prob")(self.policy_network.sample(policy_input))

        new_curr_state_q1_value = self.q1_network(torch.cat([obs_batch, new_curr_state_action,  task_z_batch.detach()],dim=1))
        new_curr_state_q2_value = self.q2_network(torch.cat([obs_batch, new_curr_state_action,  task_z_batch.detach()],dim=1))
        new_min_curr_state_q_value = torch.min(new_curr_state_q1_value, new_curr_state_q2_value)
        
        #compute policy and ent loss
        policy_loss = ((self.alpha * new_curr_state_log_pi) - new_min_curr_state_q_value).mean()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_curr_state_log_pi + self.target_entropy).detach()).mean()
            alpha_loss_value = alpha_loss.detach().cpu().item()
            self.alpha_optim.zero_grad()
        else:
            alpha_loss = 0.
            alpha_loss_value = 0.

        #backward losses, then update networks
        self.policy_optimizer.zero_grad()
        (policy_loss + alpha_loss).backward()
        self.policy_optimizer.step()
        if self.automatic_entropy_tuning:
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()
            alpha_value = self.alpha.cpu().numpy()
        else:
            alpha_value = self.alpha

        #update target v network
        self.update_target_network()
        
        kl_subject = "loss/kl_div" if self.use_information_bottleneck else "stats/kl_div"
        #exit(0)
        return {
            "loss/q1": q1_loss.item(), 
            "loss/q2": q2_loss.item(), 
            "misc/reward": reward_batch.mean().item(),
            "misc/reward_abs": torch.abs(reward_batch).mean().item(),
            "misc/target_q": target_q_value.mean().item(),
            "loss/entropy": alpha_loss_value, 
            "misc/entropy_alpha": alpha_value,
            "loss/policy": policy_loss.item(), 
            kl_subject: kl_loss.item() if self.use_information_bottleneck else kl_div.item(), 
            **{
                f"misc/train_z_mean/{train_task_indices[i]}":torch.abs(z_means[i]).mean().item() 
                    for i in range(len(train_task_indices)) if train_task_indices[i] < 5
            }, 
            **{
                f"misc/train_z_var/{train_task_indices[i]}":torch.abs(z_vars[i]).mean().item()
                    for i in range(len(train_task_indices)) if train_task_indices[i] < 5
            }
        }


    def infer_z_posterior(self, context_batch):
        z_params = self.context_encoder_network(context_batch)
        num_tasks, batch_size, _ = context_batch.shape
        z_params = z_params.view(num_tasks, batch_size, -1)
        if self.use_information_bottleneck:
            z_mean = z_params[..., :self.latent_dim]
            z_sigma_squared = nn.functional.softplus(z_params[..., self.latent_dim:])
            z_params = [product_of_gaussians(mean, std) for mean, std in zip(torch.unbind(z_mean), torch.unbind(z_sigma_squared))]
            z_means = torch.stack([p[0] for p in z_params])
            z_vars = torch.stack([p[1] for p in z_params])
        else:
            z_means = torch.mean(z_params, dim=1)
            z_vars = torch.zeros_like(z_means)
        return z_means, z_vars
 

    def sample_z_from_posterior(self, z_means, z_vars):
        if self.use_information_bottleneck:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
            z = [d.rsample() for d in posteriors]
            z = torch.stack(z)
        else:
            z = z_means
        z = z.to(util.device)
        return z 

    def update_target_network(self):
        functional.soft_update_network(self.q1_network, self.target_q1_network, self.target_smoothing_tau)
        functional.soft_update_network(self.q2_network, self.target_q2_network, self.target_smoothing_tau)

    @torch.no_grad()  
    def select_action(self, state, z, deterministic=False):
        if type(state) != torch.tensor:
            state = torch.FloatTensor(np.array([state])).to(util.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        policy_network_input = torch.cat([state, z], dim=1)
        action_info= self.policy_network.sample(policy_network_input, deterministic=deterministic)
        action = action_info['action_scaled']
        log_prob = action_info['log_prob']
        return {
            "action":action.detach().cpu().numpy()[0],
            "log_prob": log_prob
            }


class OriginalPEARLAgent(BaseAgent):
    def __init__(self, observation_space, action_space,
        reward_scale,
        target_smoothing_tau,
        kl_lambda,
        policy_mean_reg_weight,
        policy_std_reg_weight,
        policy_pre_activation_weight,
        **kwargs):
        super(OriginalPEARLAgent, self).__init__()
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        #save parameters
        self.args = kwargs

        #initilze networks
        self.latent_dim = kwargs['latent_dim']
        self.q1_network = SequentialNetwork(obs_dim + action_dim + self.latent_dim, 1, **kwargs['q_network'])
        self.q2_network = SequentialNetwork(obs_dim + action_dim + self.latent_dim, 1,**kwargs['q_network'])
        self.v_network = SequentialNetwork(obs_dim + self.latent_dim, 1,**kwargs['v_network'])
        self.target_v_network = SequentialNetwork(obs_dim + self.latent_dim, 1,**kwargs['v_network'])
        self.policy_network = TanhGaussianPolicy(obs_dim + self.latent_dim, action_space, **kwargs['policy_network'])
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']
        if self.use_next_obs_in_context:
            context_encoder_input_dim = 2 * obs_dim + action_dim + 1
        else:
            context_encoder_input_dim =  obs_dim + action_dim + 1
        self.use_information_bottleneck = kwargs['use_information_bottleneck']
        if self.use_information_bottleneck:
            context_encoder_output_dim = kwargs['latent_dim'] * 2
        else:
            context_encoder_output_dim = kwargs['latent_dim']
        self.context_encoder_network = MLPNetwork(context_encoder_input_dim, context_encoder_output_dim, **kwargs['context_encoder_network'])
        
        functional.soft_update_network(self.v_network, self.target_v_network, 1.0)

        #pass to util.device
        self.q1_network = self.q1_network.to(util.device)
        self.q2_network = self.q2_network.to(util.device)
        self.v_network = self.v_network.to(util.device)
        self.target_v_network = self.target_v_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)
        self.context_encoder_network = self.context_encoder_network.to(util.device)
        
        #register networks
        self.networks = {
            'q1_network': self.q1_network,
            'q2_network': self.q2_network,
            "v_network": self.v_network,
            "target_v_network": self.target_v_network,
            'policy_network': self.policy_network,
            'context_encoder_network': self.context_encoder_network
        }

        #initialize optimizer
        self.q1_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q1_network, kwargs['q_network']['learning_rate'])
        self.q2_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q2_network, kwargs['q_network']['learning_rate'])
        self.v_optimizer = get_optimizer(kwargs['v_network']['optimizer_class'], self.v_network, kwargs['v_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network, kwargs['policy_network']['learning_rate'])
        self.context_encoder_optimizer = get_optimizer(kwargs['context_encoder_network']['optimizer_class'], self.context_encoder_network, kwargs['context_encoder_network']['learning_rate'])

        #hyper-parameters
        self.gamma = kwargs['gamma']
        self.reward_scale = reward_scale
        self.kl_lambda = kl_lambda
        self.target_smoothing_tau = target_smoothing_tau
        self.policy_mean_reg_weight=policy_mean_reg_weight
        self.policy_std_reg_weight=policy_std_reg_weight
        self.policy_pre_activation_weight=policy_pre_activation_weight

        
    def update(self, train_task_indices, context_batch, data_batch):
        
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = data_batch
        
        # infer z from context
        z_means, z_vars = self.infer_z_posterior(context_batch)
        task_z_batch = self.sample_z_from_posterior(z_means, z_vars)
        # expand z to concatenate with obs, action batches
        num_tasks, batch_size, obs_dim = obs_batch.shape

        #flatten obs
        obs_batch = obs_batch.view(num_tasks * batch_size, -1)
        action_batch = action_batch.view(num_tasks * batch_size, -1)
        next_obs_batch = next_obs_batch.view(num_tasks * batch_size, -1)
        reward_batch = reward_batch.view(num_tasks * batch_size, -1) * self.reward_scale
        done_batch = done_batch.view(num_tasks * batch_size, -1)
        
        #expand z to match obs batch
        task_z_batch = torch.stack([task_z.repeat(batch_size, 1) for task_z in task_z_batch]).view(num_tasks * batch_size, -1)
        
        #compute Q loss and context encoder loss
        curr_state_q1_value = self.q1_network(torch.cat([obs_batch, action_batch, task_z_batch], dim=1))
        curr_state_q2_value = self.q2_network(torch.cat([obs_batch, action_batch, task_z_batch], dim=1))
        with torch.no_grad():
            next_state_v_value = self.target_v_network(torch.cat([next_obs_batch, task_z_batch], dim=1))
            target_q_value = reward_batch + (1.0 - done_batch) * self.gamma * next_state_v_value

        #compute q loss
        q1_loss = F.mse_loss(curr_state_q1_value, target_q_value)
        q2_loss =  F.mse_loss(curr_state_q2_value, target_q_value)
        q_loss = q1_loss + q2_loss

        #compute kl loss if use information bottleneck for context optimizer
        prior = torch.distributions.Normal(torch.zeros(self.latent_dim, device=util.device), torch.ones(self.latent_dim, device=util.device))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(z_means), torch.unbind(z_vars))] #todo: inpect std
        kl_divs = torch.stack([torch.distributions.kl.kl_divergence(post, prior) for post in posteriors])
        kl_div = torch.sum(kl_divs)
        kl_loss = self.kl_lambda * kl_div

        # update context encoder and q network
        self.context_encoder_optimizer.zero_grad()
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_loss.backward(retain_graph=True)
        q_loss.backward()
        self.context_encoder_optimizer.step()
        self.q1_optimizer.step()
        self.q2_optimizer.step()


        # compute v loss
        policy_input = torch.cat([obs_batch, task_z_batch.detach()], dim=1)
        policy_outputs = self.policy_network(policy_input,reparameterize=True, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        new_curr_obs_q1_value = self.q1_network(torch.cat([obs_batch, new_actions, task_z_batch.detach()], dim=1))
        new_curr_obs_q2_value = self.q2_network(torch.cat([obs_batch, new_actions, task_z_batch.detach()], dim=1))
        min_q_new_actions = torch.min(new_curr_obs_q1_value, new_curr_obs_q2_value)
        curr_obs_v_value = self.v_network(torch.cat([obs_batch, task_z_batch.detach()], dim=1))
        v_target = min_q_new_actions - log_pi
        v_loss = F.mse_loss(curr_obs_v_value, v_target.detach())

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        self.update_target_network()

        # compute loss w.r.t policy
        #compute policy and ent loss
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        kl_subject = "loss/kl_div" if self.use_information_bottleneck else "stats/kl_div"
        #exit(0)
        return {
            "loss/q1": q1_loss.item(), 
            "loss/q2": q2_loss.item(), 
            "loss/v": v_loss.item(),
            "loss/mean_reg": mean_reg_loss.item(),
            "loss/std_reg": std_reg_loss.item(),
            "loss/pre_activation_reg": pre_activation_reg_loss.item(),
            "loss/policy": policy_loss.item(), 
            "misc/reward": reward_batch.mean().item(),
            "misc/reward_abs": torch.abs(reward_batch).mean().item(),
            "misc/target_q": target_q_value.mean().item(),
            kl_subject: kl_loss.item() if self.use_information_bottleneck else kl_div.item(), 
            **{
                f"misc/train_z_mean/{train_task_indices[i]}":torch.abs(z_means[i]).mean().item() 
                    for i in range(len(train_task_indices)) if train_task_indices[i] < 5
            }, 
            **{
                f"misc/train_z_var/{train_task_indices[i]}":torch.abs(z_vars[i]).mean().item()
                    for i in range(len(train_task_indices)) if train_task_indices[i] < 5
            }
        }


    def infer_z_posterior(self, context_batch):
        z_params = self.context_encoder_network(context_batch)
        num_tasks, batch_size, _ = context_batch.shape
        z_params = z_params.view(num_tasks, batch_size, -1)
        if self.use_information_bottleneck:
            z_mean = z_params[..., :self.latent_dim]
            z_sigma_squared = nn.functional.softplus(z_params[..., self.latent_dim:])
            z_params = [product_of_gaussians(mean, std) for mean, std in zip(torch.unbind(z_mean), torch.unbind(z_sigma_squared))]
            z_means = torch.stack([p[0] for p in z_params])
            z_vars = torch.stack([p[1] for p in z_params])
        else:
            z_means = torch.mean(z_params, dim=1)
            z_vars = torch.zeros_like(z_means)
        return z_means, z_vars
 

    def sample_z_from_posterior(self, z_means, z_vars):
        if self.use_information_bottleneck:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
            z = [d.rsample() for d in posteriors]
            z = torch.stack(z)
        else:
            z = z_means
        z = z.to(util.device)
        return z 

    def update_target_network(self):
        functional.soft_update_network(self.v_network, self.target_v_network, self.target_smoothing_tau)

    @torch.no_grad()  
    def select_action(self, obs, z, deterministic=False):
        if type(obs) != torch.tensor:
            obs = torch.FloatTensor(np.array([obs])).to(util.device)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        policy_input = torch.cat([obs, z], dim=1)
        policy_outputs = self.policy_network(policy_input,reparameterize=True, return_log_prob=True, deterministic=deterministic)

        action = policy_outputs[0].detach().cpu().numpy()[0]
        return {
            "action": action,
            "log_prob": 1
            }