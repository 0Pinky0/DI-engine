from easydict import EasyDict

cartpole_sqil_config = dict(
    env_id=0,
    exp_name='cartpole_sqil_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=20,
    ),
    policy=dict(
        cuda=True,
        algorithm='sqil',
        action_space='discrete',
        model=dict(
            obs_shape=11,  # 1376
            action_shape=1657,  # 26   1657 469
            encoder_hidden_size_list=[64, 128, 128],
            dueling=True,
            # critic_head_hidden_size=128,
            # actor_head_hidden_size=128,
        ),
        learn=dict(
            epoch_per_collect=5,
            batch_size=800,
            learning_rate=0.0025,
            value_weight=0.05,  # 0.5
            entropy_weight=0.01,
            clip_ratio=0.1,
            learner=dict(hook=dict(save_ckpt_after_iter=100)),
        ),
        collect=dict(
            n_sample=1000 * 5,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=10, ), ),
    ),
)
cartpole_sqil_config = EasyDict(cartpole_sqil_config)
main_config = cartpole_sqil_config
cartpole_sqil_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sql'),
)
cartpole_sqil_create_config = EasyDict(cartpole_sqil_create_config)
create_config = cartpole_sqil_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial_sqil -c cartpole_sqil_config.py -s 0`
    # then input the config you used to generate your expert model in the path mentioned above
    # e.g. spaceinvaders_dqn_config.py
    from ding.entry import serial_pipeline_sqil
    from dizoo.classic_control.cartpole.config import cartpole_dqn_config, cartpole_dqn_create_config

    expert_main_config = cartpole_dqn_config
    expert_create_config = cartpole_dqn_create_config
    serial_pipeline_sqil((main_config, create_config), (expert_main_config, expert_create_config), seed=0)
