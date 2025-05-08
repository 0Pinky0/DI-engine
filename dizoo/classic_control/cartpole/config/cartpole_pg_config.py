from easydict import EasyDict

cartpole_pg_config = dict(
    env_id=1,
    exp_name='cartpole_pg_seed0',
    
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=195,
    ),
    policy=dict(
        algorithm = 'pg',
        cuda=False,
        random_collect_size = 0,
        model=dict(
            obs_shape=15,
            action_shape=1657,
        ),
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
            entropy_weight=0.001,
            update_per_collect= 5,
        ),
        collect=dict(unroll_len=1, 
                     discount_factor=0.9, 
                    #  n_sample = 1000*8,
                     n_episode=10, 
                     ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
    ),
)
cartpole_pg_config = EasyDict(cartpole_pg_config)
main_config = cartpole_pg_config
cartpole_pg_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='pg'),
    collector=dict(type='episode'),
)
cartpole_pg_create_config = EasyDict(cartpole_pg_create_config)
create_config = cartpole_pg_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c cartpole_pg_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
