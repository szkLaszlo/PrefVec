import omnisafe

env_id = 'grid'

custom_cfgs = {
    'train_cfgs': {
        'total_steps': 300000 if 'grid' in env_id else 1150000,
        'vector_env_nums': 1,
        'parallel': 1,
    },
    'algo_cfgs': {
        'steps_per_epoch': 20000,
    },
    'model_cfgs': {
        'actor_type': 'discrete', },
    'logger_cfgs': {
        'use_wandb': False,
        'use_tensorboard': True,
    },
}

agent = omnisafe.Agent('CPO', env_id, custom_cfgs=custom_cfgs)
agent.learn()
