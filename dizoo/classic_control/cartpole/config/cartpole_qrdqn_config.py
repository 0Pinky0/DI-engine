from easydict import EasyDict

cartpole_qrdqn_config = dict(
    env_id=0,
    exp_name='cartpole_qrdqn_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        algorithm = 'qrdqn_nstep',
        model=dict(
            obs_shape=11,
            action_shape=8,
            encoder_hidden_size_list=[128, 128, 64],
            num_quantiles=64,
        ),
        discount_factor=0.97,
        nstep=1,
        learn=dict(
            update_per_collect=3,
            batch_size=64,
            learning_rate=0.001,
            target_update_freq=100,
            kappa=1.0,
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
cartpole_qrdqn_config = EasyDict(cartpole_qrdqn_config)
main_config = cartpole_qrdqn_config
cartpole_qrdqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='qrdqn'),
)
cartpole_qrdqn_create_config = EasyDict(cartpole_qrdqn_create_config)
create_config = cartpole_qrdqn_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_qrdqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
