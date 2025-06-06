from easydict import EasyDict

cartpole_ppg_config = dict(
    env_id=0,
    exp_name='cartpole_ppg_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=2,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        algorithm='ppg',
        model=dict(
            obs_shape=170,
            action_shape=7,
            encoder_hidden_size_list=[64, 128, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=10,
            batch_size=256,
            learning_rate=0.00025,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
        ),
        collect=dict(
            n_sample=5120,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=20, )),
        other=dict(
            replay_buffer=dict(
                multi_buffer=True,
                policy=dict(
                    replay_buffer_size=4096,
                    max_use=10,
                ),
                value=dict(
                    replay_buffer_size=20480,
                    max_use=50,
                ),
            ),
        ),
    ),
)
cartpole_ppg_config = EasyDict(cartpole_ppg_config)
main_config = cartpole_ppg_config
cartpole_ppg_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppg_offpolicy'),
    replay_buffer=dict(
        policy=dict(type='advanced'),
        value=dict(type='advanced'),
    )
)
cartpole_ppg_create_config = EasyDict(cartpole_ppg_create_config)
create_config = cartpole_ppg_create_config

if __name__ == "__main__":
    # This config file can be executed by `dizoo/classic_control/cartpole/entry/cartpole_ppg_main.py`
    import os
    import warnings
    from dizoo.classic_control.cartpole.entry.cartpole_ppg_main import main
    from dizoo.classic_control.cartpole.entry.cartpole_ppg_main import __file__ as _origin_py_file

    origin_py_file_rel = os.path.relpath(_origin_py_file, os.path.abspath(os.path.curdir))
    warnings.warn(UserWarning(f"This config file can be executed by {repr(origin_py_file_rel)}"))
    main(cartpole_ppg_config)
