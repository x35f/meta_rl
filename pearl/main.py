import os
import sys
# sys.path.append(os.path.join(os.getcwd(), './'))
# sys.path.append(os.path.join(os.getcwd(), '../../'))
import gym
import click
from unstable_baselines.common.logger import Logger
from unstable_baselines.meta_rl.pearl.trainer import PEARLTrainer
from unstable_baselines.meta_rl.pearl.agent import PEARLAgent
from unstable_baselines.common.util import set_device_and_logger, load_config, set_global_seed
from unstable_baselines.common.buffer import ReplayBuffer
from unstable_baselines.common.env_wrapper import get_env


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("config-path",type=str, default="sac/configs/default_with_per.json")
@click.option("--log-dir", default=os.path.join("logs", "pearl"))
@click.option("--gpu", type=int, default=-1)
@click.option("--print-log", type=bool, default=True)
@click.option("--seed", type=int, default=35)
@click.option("--info", type=str, default="")
@click.option("--load-dir", type=str, default="")
@click.argument('args', nargs=-1)
def main(config_path, log_dir, gpu, print_log, seed, info, load_dir, args):
    #todo: add load and update parameters function
    args = load_config(config_path, args)

    #set global seed
    set_global_seed(seed)

    #initialize logger
    env_name = args['env_name']
    logger = Logger(log_dir,env_name, prefix = info, print_to_terminal=print_log)

    #set device and logger
    set_device_and_logger(gpu, logger)

    #save args
    logger.log_str_object("parameters", log_dict = args)

    #initialize environment
    logger.log_str("Initializing Environment")
    num_train_tasks = args['common']['num_train_tasks']
    num_eval_tasks = args['common']['num_eval_tasks']
    train_env = get_env(env_name, n_tasks=num_train_tasks, randomize_tasks=True)
    eval_env = get_env(env_name, n_tasks=num_eval_tasks, randomize_tasks=True)
    state_space = train_env.observation_space
    action_space = train_env.action_space

    #initialize buffer
    logger.log_str("Initializing Buffer")
    train_replay_buffers = [ReplayBuffer(state_space, action_space, **args['replay_buffer']) for _ in range(num_train_tasks)]
    train_encoder_buffers = [ReplayBuffer(state_space, action_space, **args['encoder_buffer']) for _ in range(num_train_tasks)]
    eval_buffer = ReplayBuffer(state_space, action_space, **args['encoder_buffer'])

    #initialize agent
    logger.log_str("Initializing Agent")
    agent = PEARLAgent(state_space, action_space, **args['agent'])

    #initialize trainer
    logger.log_str("Initializing Trainer")
    trainer  = PEARLTrainer(
        agent,
        train_env,
        eval_env,
        train_replay_buffers,
        train_encoder_buffers,
        eval_buffer,
        load_dir,
        **args['trainer']
    )

    
    logger.log_str("Started training")
    trainer.train()


if __name__ == "__main__":
    main()