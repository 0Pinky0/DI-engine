from easydict import EasyDict


cartpole_pdqn_config = dict(
    env_id = 0,
    exp_name='cartpole_pdqn_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        # (bool) Scale output action into legal range [-1, 1].
        act_scale=True,
        env_id='Moving-v0',  # ['Sliding-v0', 'Moving-v0']
        n_evaluator_episode=5,
        stop_value=1.8,
    ),
    policy=dict(
        algorithm = 'pdqn',
        cuda=True,
        discount_factor=0.99,
        nstep=1,
        model=dict(
            obs_shape=11,
            action_shape=8,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=False,
        ),
        learn=dict(
            update_per_collect=50,  # 10~500
            batch_size=320,
            learning_rate_dis=3e-4,
            learning_rate_cont=3e-4,
            target_theta=0.001,
            update_circle=10,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            n_sample=3200,  # 128,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            noise_sigma=0.1,  # 0.05,
            collector=dict(collect_print_freq=100, ),
        ),
        eval=dict(evaluator=dict(eval_freq=50, ), ),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                start=1,
                end=0.1,
                # (int) Decay length(env step)
                decay=int(1e5),
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), ),
        ),
    )
)

cartpole_pdqn_config = EasyDict(cartpole_pdqn_config)
main_config = cartpole_pdqn_config

cartpole_pdqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='pdqn'),
)
cartpole_pdqn_create_config = EasyDict(cartpole_pdqn_create_config)
create_config = cartpole_pdqn_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c gym_hybrid_pdqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0, dynamic_seed=False, max_env_step=int(1e7))
