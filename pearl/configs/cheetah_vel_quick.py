overwrite_args = {
  "env_name": "cheetah-vel",
  "common":{
    "num_train_tasks": 10,
    "num_eval_tasks": 5
  },
  "trainer":{
    "max_epoch": 200,
    "z_inference_batch_size": 100,
    "num_eval_exploration_trajectories":2
  } 
}
