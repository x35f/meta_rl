overwrite_args = {
  "env_name": "ant-dir",
  "common":{
    "num_train_tasks": 2,
    "num_eval_tasks": 2,
    "use_same_tasks_for_eval": True
  },
  "trainer":{
    "num_train_tasks_to_sample_per_epoch": 4,
    "z_inference_batch_size": 256,
    "num_steps_prior": 400, 
    "num_steps_posterior": 0, 
    "num_extra_steps_posterior": 600,
    "num_eval_exploration_trajectories": 2,
    "num_updates_per_epoch":400,
    "num_eval_rounds": 4 
  } 
}
