from easydict import EasyDict

cartpole_dqn_config = dict(
    env_id=0,
    exp_name='wargame_dqn_her_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=5000,
        # replay_path='cartpole_dqn_seed0/video',
    ),
    policy=dict(
        cuda=True,
        algorithm='dqn',
        load_path='models/wargame_dqn_her_seed0/ckpt/ckpt_best.pth.tar',  # necessary for eval
        model=dict(
            obs_shape=11,
            action_shape=7,
            encoder_hidden_size_list=[64, 128, 128],
            dueling=True,
            # dropout=0.1,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            update_per_collect=150,
            batch_size=2000,
            learning_rate=0.00001,
        ),
        collect=dict(
            n_sample=2000 * 2,
            #  n_episode = 3
        ),
        eval=dict(evaluator=dict(eval_freq=50, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=20000, ),
            # her=dict(
            #     her_strategy='future',
            #     # her_replay_k=2,  # `her_replay_k` is not used in episodic HER
            #     # Sample how many episodes in each train iteration.
            #     episode_size=32,
            #     # Generate how many samples from one episode.
            #     sample_per_episode=4,
            # ),
        ),
    ),
)
cartpole_dqn_config = EasyDict(cartpole_dqn_config)
main_config = cartpole_dqn_config
cartpole_dqn_create_config = dict(
    # env=dict(
    #     type='wargame',
    #     import_names=['wargame.scout_env.train_env'],
    # ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
    replay_buffer=dict(type='deque', import_names=['ding.data.buffer.deque_buffer_wrapper']),
)
cartpole_dqn_create_config = EasyDict(cartpole_dqn_create_config)
create_config = cartpole_dqn_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_dqn_config.py -s 0`
    from ding.entry import serial_pipeline

    serial_pipeline((main_config, create_config), seed=0)
