overwrite_args = {
  "env_name": "walker-rand-params",
  "common":{
    "use_next_obs_in_context": True,
    "num_train_tasks": 40,
    "num_test_tasks": 10
  },
  "trainer":{
    "num_train_tasks_to_sample_per_epoch": 10,
    "z_inference_batch_size": 100,
    "num_steps_prior": 400, 
    "num_steps_posterior": 0, 
    "num_extra_steps_posterior": 600,
    "num_updates_per_epoch":4000,
    "num_eval_exploration_trajectories": 2
  }
}
