overwrite_args = {
  "env_name": "ant-goal",
  "use_new_sac": False,
  "common":{
    "num_train_tasks": 150,
    "num_eval_tasks": 30
  },
  "agent":{
    "kl_lambda": 1.0
  },
  "trainer":{
    "num_train_tasks_to_sample_per_epoch": 10,
    "z_inference_batch_size": 256,
    "num_steps_prior": 400, 
    "num_steps_posterior": 0, 
    "num_extra_steps_posterior": 600,
    "num_eval_exploration_trajectories": 2,
    "num_updates_per_epoch":4000
  } 
}
