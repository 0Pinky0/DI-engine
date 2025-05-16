from easydict import EasyDict

cartpole_dqn_config = dict(
    env_id=0,
    exp_name='miaosuan_dqn_rnd_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=195,
    ),
    reward_model=dict(
        intrinsic_reward_type='add',
        learning_rate=1e-3,
        obs_shape=170,
        batch_size=256,
        update_per_collect=10,
    ),
    policy=dict(
        cuda=True,
        algorithm='dqn_rnd',
        model=dict(
            obs_shape=170,
            action_shape=7,
            encoder_hidden_size_list=[64, 128, 128],
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            batch_size=64,
            learning_rate=0.0001,
            update_per_collect=5,
        ),
        collect=dict(n_sample=5120),
        eval=dict(evaluator=dict(eval_freq=10, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=100000,
            ),
            replay_buffer=dict(replay_buffer_size=51200, ),
        ),
    ),
)
cartpole_dqn_config = EasyDict(cartpole_dqn_config)
main_config = cartpole_dqn_config
cartpole_dqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
    replay_buffer=dict(type='deque'),
    reward_model=dict(type='rnd'),
)
cartpole_dqn_create_config = EasyDict(cartpole_dqn_create_config)
create_config = cartpole_dqn_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_dqn_config.py -s 0`
    from ding.entry import serial_pipeline

    serial_pipeline((main_config, create_config), seed=0)
