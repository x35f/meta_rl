import torch
from torch._C import device
import torch.nn as nn
from torch.distributions import Categorical, Normal
import numpy as np
import gym
from gym.spaces import Discrete, Box, MultiBinary
from typing import List, Tuple, Any, Sequence, Union, final
from abc import ABC, abstractmethod

from torch.nn.modules.activation import Tanh
from unstable_baselines.common import util
from unstable_baselines.common.networks import PolicyNetwork, get_act_cls, get_network


# # TODO
# BasePolicy
# |---DeterministicPolicy
# |
# |---StochasticPolicya

class BasePolicyNetwork(ABC, nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 action_space: gym.Space, 
                 hidden_dims: Union[Sequence[int], int], 
                 act_fn: str = "relu", 
                 *args, **kwargs
        ):
        super(BasePolicyNetwork, self).__init__()

        self.input_dim = input_dim
        self.action_space = action_space
        self.args = args
        self.kwargs = kwargs
        
        if isinstance(hidden_dims,  int):
            hidden_dims = [hidden_dims]
        hidden_dims = [input_dim] + hidden_dims

        # init hidden layers
        self.hidden_layers = []
        act_cls = get_act_cls(act_fn)
        for i in range(len(hidden_dims) - 1):
            curr_shape, next_shape = hidden_dims
            curr_network = get_network([curr_shape, next_shape])
            self.hidden_layers.extend([curr_network, act_cls()])

        # init output layer shape
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
        elif isinstance(action_space, Box):
            self.action_dim = action_space.shape[0]
        elif isinstance(action_space, MultiBinary):
            self.aaction_dim = action_space.shape[0]
        else:
            raise TypeError        


    @abstractmethod
    def forward(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample(self, state, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(self, states, actions, *args, **kwargs):
        raise NotImplementedError

    def to(self, device):
        return nn.Module.to(self, device)

class DeterministicPolicyNetwork(BasePolicyNetwork):
    def __init__(self, 
                 input_dim: int, 
                 action_space: gym.Space, 
                 hidden_dims: Union[Sequence[int], int], 
                 act_fn: str = "relu", 
                 out_act_fn: str = "identity", 
                 *args, **kwargs
        ):
        super(DeterministicPolicyNetwork, self).__init__(input_dim, action_space, hidden_dims, act_fn, *args,  **kwargs)

        self.deterministic = True
        self.policy_type = "deterministic"

        # get final layer
        final_network = get_network([hidden_dims[-1], self.action_dim])
        out_act_cls = get_act_cls(out_act_fn)
        self.networks = nn.Sequential(*self.hidden_layers, final_network, out_act_cls())

        # set noise
        self.noise = torch.Tensor(self.action_dim)

        # set scaler
        if action_space is None:
            self.action_scale = nn.Parameter(torch.tensor(1., dtype=torch.float, device=util.device), requires_grad=False)
            self.action_bias = nn.Parameter(torch.tensor(0., dtype=torch.float, device=util.device), requires_grad=False)
        elif not isinstance(action_space, Discrete):
            self.action_scale = nn.Parameter(torch.tensor( (action_space.high-action_space.low)/2.0, dtype=torch.float, device=util.device), requires_grad=False)
            self.action_bias = nn.Parameter(torch.tensor( (action_space.high+action_space.low)/2.0, dtype=torch.float, device=util.device), requires_grad=False)

    def forward(self, state: torch.Tensor):
        out = self.networks(state)
        return out

    def sample(self, state: torch.Tensor):
        action_prev_tanh = self.networks(state)
        action_raw = torch.tanh(action_prev_tanh, dim=-1)
        action_scaled = action_raw * self.action_scale + self.action_bias
            
        return {
            "action_prev_tanh": action_prev_tanh,
            "action_raw": action_raw, 
            "action_scaled": action_scaled,  
        }

    # CHECK: I'm not sure about the reparameterization trick used in DDPG
    def evaluate_actions(self, state: torch.Tensor):
        action_prev_tanh = self.networks(state)
        action_raw = torch.tanh(action_prev_tanh, dim=-1)
        action_scaled = action_raw * self.action_scale + self.action_bias

        return {
            "action_prev_tanh": action_prev_tanh,
            "action_raw": action_raw, 
            "action_scaled": action_scaled,  
        }
        

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        super(DeterministicPolicyNetwork, self).to(device)


class CategoricalPolicyNetwork(BasePolicyNetwork):
    def __init__(self, 
                 input_dim: int, 
                 action_space: gym.Space, 
                 hidden_dims: Union[Sequence[int], int], 
                 act_fn: str = "relu", 
                 out_act_fn: str = "identity", 
                 *args, **kwargs
        ):
        super(CategoricalPolicyNetwork, self).__init__(input_dim, action_space, hidden_dims, act_fn, *args, **kwargs)

        self.determnistic = False
        self.policy_type = "categorical"

        # get final layer
        final_network = get_network([hidden_dims[-1], self.action_dim])
        out_act_cls = get_act_cls(out_act_fn)
        self.networks = nn.Sequential(*self.hidden_layers, final_network, out_act_cls())

        # categorical do not have scaler, and do not support re_parameterization

    def forward(self, state: torch.Tensor):
        out = self.networks(state)
        return out

    def sample(self, state: torch.Tensor, deterministic=False):
        logit = self.forward(state)
        probs = torch.softmax(logit, dim=-1)
        if deterministic:
            return {
                "logit": logit, 
                "probs": probs, 
                "action": torch.argmax(probs, dim=-1),
                "log_prob": torch.log(torch.max(probs, dim=-1) + 1e-6), 
            }
        else:
            dist = Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return {
                "logit": logit, 
                "probs": probs, 
                "action": action, 
                "log_prob": log_prob, 
            }

    def evaluate_actions(self, states, actions, *args, **kwargs):
        logit = self.forward(states)
        probs = torch.softmax(logit, dim=1)
        dist = Categorical(probs)
        return dist.log_prob(actions), dist.entropy()

    def to(self, device):
        super(CategoricalPolicyNetwork, self).to(device)


class GaussianPolicyNetwork(BasePolicyNetwork):
    def __init__(self, 
                 input_dim: int, 
                 action_space: gym.Space, 
                 hidden_dims: Union[Sequence[int], int], 
                 act_fn: str = "relu", 
                 out_act_fn: str = "identity", 
                 re_parameterize: bool = True,
                 *args, **kwargs
        ):
        super(GaussianPolicyNetwork, self).__init__(input_dim, action_space, hidden_dims, act_fn)

        self.deterministic = False
        self.policy_type = "Gaussian"

        # get final layer
        final_network = get_network([hidden_dims[-1], self.action_dim*2])
        out_act_cls = get_act_cls(out_act_fn)
        self.networks = nn.Sequential(*self.hidden_layers, final_network, out_act_cls())

        # set scaler
        if action_space is None:
            self.action_scale = nn.Parameter(torch.tensor(1., dtype=torch.float, device=util.device), requires_grad=False)
            self.action_bias = nn.Parameter(torch.tensor(0., dtype=torch.float, device=util.device), requires_grad=False)
        else:
            self.action_scale = nn.Parameter(torch.tensor( (action_space.high-action_space.low)/2.0, dtype=torch.float, device=util.device), requires_grad=False)
            self.action_bias = nn.Parameter(torch.tensor( (action_space.high+action_space.low)/2.0, dtype=torch.float, device=util.device), requires_grad=False)

        self.re_parameterize = re_parameterize

    def forward(self, state: torch.Tensor):
        out = self.networks(state)
        action_mean = out[:, :self.action_dim]
        action_log_var = out[:, self.action_dim:]
        return action_mean, action_log_var


    def sample(self, state: torch.Tensor, deterministic: bool=False):
        mean, log_var = self.forward(state)
        if deterministic:
            # action sample must be detached !
            action_prev_tanh = mean.detach()            
        else:
            dist = Normal(mean, log_var.exp())
            if self.re_parameterize:
                action_prev_tanh = dist.rsample()
            else:
                action_prev_tanh = dist.sample()

        action_raw = torch.tanh(action_prev_tanh)
        action_scaled = action_raw * self.action_scale + self.action_bias
            
        log_prob_prev_tanh = dist.log_prob(action_raw)
        log_prob = log_prob_prev_tanh - torch.log(self.action_scale*(1-torch.tanh(action_prev_tanh).pow(2)) + 1e-6)
        log_prob = torch.sum(log_prob, dim=1)
        return {
            "action_prev_tanh": action_prev_tanh, 
            "action_raw": action_raw, 
            "action_scaled": action_scaled, 
            "log_prob_prev_tanh": log_prob_prev_tanh, 
            "log_prob": log_prob
        }

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor, action_type: str = "scaled"):
        # should not be used by SAC, because SAC only replays states in buffer
        mean, log_var = self.forward(states)
        dist = Normal(mean, log_var.exp())

        if action_type == "scaled":
            actions = (actions - self.action_bias) / self.action_scale
            actions = torch.atanh(actions)
        elif action_type == "raw":
            actions = torch.atanh(actions)
        log_pi = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy()
        return log_pi, entropy

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        super(GaussianPolicyNetwork, self).to(device)







            
        
        