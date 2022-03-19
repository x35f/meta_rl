import torch
from torch.distributions import  Normal
from unstable_baselines.common.networks import GaussianPolicyNetwork
from torch.distributions import Distribution, Normal
from unstable_baselines.common import util
from unstable_baselines.common.networks import get_network, get_act_cls
from torch.autograd import Variable
import torch.nn as nn
from gym.spaces import Discrete, Box, MultiBinary, space
import numpy as np
import torch.nn.functional as F

class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        z = self.normal.sample()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        z = (
            self.normal_mean +
            self.normal_std *
            Variable(Normal(
                torch.zeros(self.normal_mean.size(), device=util.device),
                torch.ones(self.normal_std.size(), device=util.device)
            ).sample())
        )
        # z.requires_grad_()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)
            

            z = mean +std *sample


def identity(x):
    return x

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

class Mlp(torch.nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=fanin_init,
            b_init_value=0.1,
    ):
        super().__init__()


        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class TanhGaussianPolicy(Mlp):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            latent_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.latent_dim = latent_dim
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions[0, :], {}

    def forward(
            self,
            obs,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
class GaussianPolicyNetwork(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 action_space ,
                 hidden_dims , 
                 act_fn: str = "relu", 
                 out_act_fn: str = "identity", 
                 re_parameterize: bool = True,
                 fix_std: bool = False,
                 paramterized_std: bool = False,
                 log_std: float = None,
                 log_std_min: int = -20, 
                 log_std_max: int = 2, 
                 stablize_log_prob: bool=True,
                **kwargs
        ):
        super(GaussianPolicyNetwork, self).__init__()
        self.input_dim = input_dim
        self.action_space = action_space
        
        if isinstance(hidden_dims,  int):
            hidden_dims = [hidden_dims]
        hidden_dims = [input_dim] + hidden_dims

        # init hidden layers
        self.hidden_layers = []
        act_cls = get_act_cls(act_fn)
        for i in range(len(hidden_dims) - 1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i+1]
            curr_network = get_network([curr_shape, next_shape])
            self.hidden_layers.extend([curr_network, act_cls()])

        # init output layer shape
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
        elif isinstance(action_space, Box):
            self.action_dim = action_space.shape[0]
        elif isinstance(action_space, MultiBinary):
            self.action_dim = action_space.shape[0]
        else:
            raise TypeError        
        self.deterministic = False
        self.policy_type = "Gaussian"
        self.fix_std = fix_std
        self.re_parameterize = re_parameterize

        # get final layer
        if not self.fix_std:
            final_network = get_network([hidden_dims[-1], self.action_dim * 2])
        else:
            final_network = get_network([hidden_dims[-1], self.action_dim])
        out_act_cls = get_act_cls(out_act_fn)
        self.networks = nn.Sequential(*self.hidden_layers, final_network, out_act_cls())

        # set scaler
        if action_space is None:
            self.action_scale = nn.Parameter(torch.tensor(1., dtype=torch.float, device=util.device), requires_grad=False)
            self.action_bias = nn.Parameter(torch.tensor(0., dtype=torch.float, device=util.device), requires_grad=False)
        else:
            self.action_scale = nn.Parameter(torch.tensor( (action_space.high-action_space.low)/2.0, dtype=torch.float, device=util.device), requires_grad=False)
            self.action_bias = nn.Parameter(torch.tensor( (action_space.high+action_space.low)/2.0, dtype=torch.float, device=util.device), requires_grad=False)


        # set log_std
        if log_std == None:
            self.log_std = -0.5 * np.ones(self.action_dim, dtype=np.float32)
        else:
            self.log_std = log_std
        if paramterized_std:
            self.log_std = torch.nn.Parameter(torch.as_tensor(self.log_std)).to(util.device)
        else:
            self.log_std = torch.tensor(self.log_std, dtype=torch.float, device=util.device)
        self.log_std_min = nn.Parameter(torch.tensor(log_std_min, dtype=torch.float, device=util.device ), requires_grad=False)
        self.log_std_max = nn.Parameter(torch.tensor(log_std_max, dtype=torch.float, device=util.device ), requires_grad=False)
        self.stablize_log_prob = stablize_log_prob

    def forward(self, obs: torch.Tensor):
        out = self.networks(obs)
        action_mean = out[:, :self.action_dim]
        # check whether the `log_std` is fixed in forward() to make the sample function
        # keep consistent
        if self.fix_std:
            action_log_std = self.log_std
        else:
            action_log_std = out[:, self.action_dim:]       
        return action_mean, action_log_std

    def sample(self, obs: torch.Tensor, deterministic: bool=False):

        mean, log_std = self.forward(obs)
        # util.debug_print(type(log_std), info="Gaussian Policy sample")
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
       
        log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            action, pre_tanh_value = tanh_normal.rsample(
                return_pretanh_value=True
            )
            log_prob = tanh_normal.log_prob(
                action,
                pre_tanh_value=pre_tanh_value
            )
            log_prob = log_prob.sum(dim=1, keepdim=True)
    
        return {
            "action_raw_mean": mean, 
            "action_scaled": action,
            "log_prob": log_prob,
        }

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor, action_type: str = "scaled"):
        """ Evaluate action to get log_prob and entropy.
        
        Note: This function should not be used by SAC because SAC only replay states in buffer.
        """
        mean, log_std = self.forward(states)
        dist = Normal(mean, log_std.exp())

        if action_type == "scaled":
            actions = (actions - self.action_bias) / self.action_scale
            actions = torch.atanh(actions)
        elif action_type == "raw":
            # actions = torch.atanh(actions)
            pass
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return {
            "log_prob": log_prob,
            "entropy": entropy
            }

    def to(self, device):
        self.action_scale = self.action_scale.to(util.device)
        self.action_bias = self.action_bias.to(util.device)
        return super(GaussianPolicyNetwork, self).to(device)

