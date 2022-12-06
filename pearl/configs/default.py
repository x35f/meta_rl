default_args = {
  "env_name": "cheetah-dir",
  "use_new_sac": True,
  "env":{
    "randomize_tasks": True
  },
  "common":{
    "use_next_obs_in_context": False, # whether to use next observation in context
    "num_train_tasks": 2,
    "num_eval_tasks": 2,
    "use_same_tasks_for_eval": False
  },
  "replay_buffer":{
    "max_buffer_size": 1000000
  },
  "encoder_buffer":{
    "max_buffer_size": 100000
  },
  "agent":{
    "gamma": 0.99,
    "reward_scale": 5.0,
    "target_smoothing_tau": 0.005,
    "latent_dim": 5,
    "use_information_bottleneck": True,
    "kl_lambda": 0.1,
    "alpha": 0.2,                             # entropy alpha for new pearl agent
    "policy_mean_reg_weight":1e-3,            # for original pearl agent
    "policy_std_reg_weight":1e-3,             # for original pearl agent
    "policy_pre_activation_weight":0.0,       # for original pearl agent
    "q_network":{
      "network_params": [("mlp", 300), ("mlp", 300), ("mlp", 300)],
      "optimizer_class": "Adam",
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity"
    },
    "v_network":{
      "network_params": [("mlp", 300), ("mlp", 300), ("mlp", 300)],
      "optimizer_class": "Adam",
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity"
    },
    "policy_network":{
      "network_params": [("mlp", 300), ("mlp", 300), ("mlp", 300)],
      "optimizer_class": "Adam",
      "deterministic": False,
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity",
      "re_parameterize": True,
      "stablize_log_prob": True
    },
    "context_encoder_network":{
      "network_params": [("mlp", 200), ("mlp", 200), ("mlp", 200)],
      "optimizer_class": "Adam",
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity"
    },
    "entropy":{                               # for new pearl agent
      "automatic_tuning": True,
      "learning_rate": 0.0003,
      "optimizer_class": "Adam"
    }
  },
  "trainer":{
    "batch_size": 256,                        # batch size for training
    "z_inference_batch_size": 64,             # number of transitions required to infer z
    "max_epoch": 500,                         # max training epoch
    "num_tasks_per_gradient_update": 16,      # number of tasks to sample per gradient update
    "num_train_tasks_to_sample_per_epoch": 5, # number of tasks to collect data for each epoch
    "num_steps_prior": 400,                   # number of prior samples to collect for agent and encoder training       
    "num_steps_posterior": 0,                 # number of posterior samples to collect for agent and encoder training
    "num_extra_steps_posterior": 400,         # number of extra posterior samples to collect for agent training only
    "num_updates_per_epoch":2000,             # number of gradient updates to perform per epoch
    "adaptation_context_update_interval": 1,  # how often to update context during adaptation phase
    "max_trajectory_length":200,
    "eval_interval": 1,                       # how often to evaluate (in epoch)
    "num_eval_trajectories": 5,
    "num_eval_exploration_trajectories":1,    # number of exploration trajectories during evalutation (before infering z)
    "num_eval_rounds": 2,                     # number of evaluation rounds to perform
    "snapshot_interval": 10,
    "num_initial_steps": 2000,                # number of initial steps per task to collect before training starts
    "save_video_demo_interval": -1,
    "log_interval": 1
  }
}