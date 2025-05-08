from easydict import EasyDict

cartpole_rainbow_config = dict(
    env_id =0,
    exp_name='cartpole_rainbow_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=195,
    ),
    policy=dict(
        cuda=True,
        priority=True,
        algorithm = 'rainbow',
        discount_factor=0.97,
        nstep=1,
        model=dict(
            obs_shape=11,
            action_shape=8,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            update_per_collect=3,
            batch_size=64,
            learning_rate=0.001,
            target_update_freq=100,
        ),
        collect=dict(
            n_sample=800*5,
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=10, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ), replay_buffer=dict(replay_buffer_size=20000, )
        ),
    ),
)
cartpole_rainbow_config = EasyDict(cartpole_rainbow_config)
main_config = cartpole_rainbow_config
cartpole_rainbow_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='rainbow'),
)
cartpole_rainbow_create_config = EasyDict(cartpole_rainbow_create_config)
create_config = cartpole_rainbow_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_rainbow_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
