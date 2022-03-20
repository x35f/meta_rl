overwrite_args = {
  "env_name": "humanoid-dir",
  "common":{
    "num_train_tasks": 100,
    "num_eval_tasks": 30
  },
  "trainer":{
    "z_inference_batch_size": 256,
    "num_steps_prior": 400, 
    "num_steps_posterior": 0, 
    "num_extra_steps_posterior": 600,
    "num_eval_exploration_trajectories": 2,
    "num_updates_per_epoch":4000
  } 
}
