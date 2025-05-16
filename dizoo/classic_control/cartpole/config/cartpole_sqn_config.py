from easydict import EasyDict

update_per_collect = 8
cartpole_sqn_config = dict(
    env_id=0,
    exp_name='cartpole_sqn_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=5000,
    ),
    policy=dict(
        cuda=True,
        algorithm='sqn',
        action_space='discrete',
        model=dict(
            obs_shape=170,
            action_shape=7,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
            # dropout=0.1,
        ),
        learn=dict(
            multi_gpu=False,
            update_per_collect=update_per_collect,
            batch_size=128,
            learning_rate_q=0.001,
            learning_rate_alpha=0.001,
            alpha=0.2,
            target_entropy=0.2,
        ),
        collect=dict(
            n_sample=1024,
            nstep=1,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.2,
                decay=2000,
            ), replay_buffer=dict(replay_buffer_size=10240, )
        ),
        eval=dict(evaluator=dict(eval_freq=5, ), ),
    )
)
cartpole_sqn_config = EasyDict(cartpole_sqn_config)
main_config = cartpole_sqn_config

cartpole_sqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sqn'),
)
cartpole_sqn_create_config = EasyDict(cartpole_sqn_create_config)
create_config = cartpole_sqn_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_sqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
