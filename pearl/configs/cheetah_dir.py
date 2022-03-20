overwrite_args = {
  "env_name": "cheetah-dir",
  "common":{
    "num_train_tasks": 2,
    "num_test_tasks": 2,
    "use_same_tasks_for_eval": True
  },
  "trainer":{
    "num_train_tasks_to_sample_per_epoch": 4,
    "z_inference_batch_size": 256,
    "num_steps_prior": 1000, 
    "num_steps_posterior": 0, 
    "num_extra_steps_posterior": 1000,
    "num_eval_exploration_trajectories": 2,
    "num_eval_rounds": 4
  }
}
