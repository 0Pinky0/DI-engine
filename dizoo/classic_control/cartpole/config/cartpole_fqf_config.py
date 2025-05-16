from easydict import EasyDict

cartpole_fqf_config = dict(
    env_id=0,
    exp_name='cartpole_fqf_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=195,
        replay_path='cartpole_fqf_seed0/video',
    ),
    policy=dict(
        cuda=True,
        algorithm='fqf',
        priority=True,
        model=dict(
            obs_shape=170,
            action_shape=7,
            encoder_hidden_size_list=[128, 128, 64],
            num_quantiles=32,
            quantile_embedding_size=64,
        ),
        discount_factor=0.97,
        nstep=1,
        learn=dict(
            update_per_collect=3,
            batch_size=64,
            learning_rate_fraction=0.0001,
            learning_rate_quantile=0.0001,
            target_update_freq=50,
            ent_coef=1,
        ),
        collect=dict(
            n_sample=1024,
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=10, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=100000,
            ), replay_buffer=dict(replay_buffer_size=10240, )
        ),
    ),
)
cartpole_fqf_config = EasyDict(cartpole_fqf_config)
main_config = cartpole_fqf_config
cartpole_fqf_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='fqf'),
)
cartpole_fqf_create_config = EasyDict(cartpole_fqf_create_config)
create_config = cartpole_fqf_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c cartpole_fqf_config.py -s 0`
    from ding.entry import serial_pipeline

    serial_pipeline((main_config, create_config), seed=0)
