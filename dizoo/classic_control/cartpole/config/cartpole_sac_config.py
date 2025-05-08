from easydict import EasyDict

cartpole_sac_config = dict(
    env_id=0,
    exp_name='cartpole_sac_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=0,
        algorithm = 'sac',
        load_path='models/wargame_sac_seed0/ckpt/ckpt_best.pth.tar',
        multi_agent=False,
        model=dict(
            obs_shape=11,
            action_shape=8,
            twin_critic=True,
            actor_head_hidden_size=64,
            critic_head_hidden_size=64,
        ),
        learn=dict(
            update_per_collect=2,
            batch_size=64,
            learning_rate_q=5e-3,
            learning_rate_policy=5e-3,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.01,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=False,
        ),
        collect=dict(
            env_num=8,
            n_sample=800*5,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(
            evaluator=dict(eval_freq=10, ),
            env_num=5,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=50000,
            ), replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
)

cartpole_sac_config = EasyDict(cartpole_sac_config)
main_config = cartpole_sac_config

cartpole_sac_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='discrete_sac'),
)
cartpole_sac_create_config = EasyDict(cartpole_sac_create_config)
create_config = cartpole_sac_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_sac_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
