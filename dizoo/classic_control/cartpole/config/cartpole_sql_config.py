from easydict import EasyDict

cartpole_sql_config = dict(
    env_id=0,
    exp_name='cartpole_sql_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=5000,
    ),
    policy=dict(
        cuda=True,
        algorithm='sql',
        action_space='discrete',
        model=dict(
            obs_shape=170,  # 1376
            action_shape=7,  # 26   1657 469
            encoder_hidden_size_list=[64, 128, 128],
            dueling=True,
            # critic_head_hidden_size=128,
            # actor_head_hidden_size=128,
        ),
        learn=dict(
            epoch_per_collect=5,
            batch_size=512,
            learning_rate=0.0025,
            value_weight=0.05,  # 0.5
            entropy_weight=0.01,
            clip_ratio=0.1,
            learner=dict(hook=dict(save_ckpt_after_iter=40)),
        ),
        collect=dict(
            n_sample=5120,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=20, ), ),
    ),
)
cartpole_sql_config = EasyDict(cartpole_sql_config)
main_config = cartpole_sql_config
cartpole_sql_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sql'),
)
cartpole_sql_create_config = EasyDict(cartpole_sql_create_config)
create_config = cartpole_sql_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_sql_config.py -s 0`
    from ding.entry import serial_pipeline

    serial_pipeline((main_config, create_config), seed=0)
