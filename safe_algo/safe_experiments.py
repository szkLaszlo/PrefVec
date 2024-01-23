import omnisafe


env_id = 'CartPole-v1'
custom_cfgs = {
    'train_cfgs': {
        'total_steps': 10000000,
        'vector_env_nums': 1,
        'parallel': 1,
    },
    'algo_cfgs': {
        'steps_per_epoch': 20000,
    },
    'logger_cfgs': {
        'use_wandb': False,
        'use_tensorboard': True,
    },
}

agent = omnisafe.Agent('CPO', env_id, custom_cfgs=custom_cfgs)
agent.learn()