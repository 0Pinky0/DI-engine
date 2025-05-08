from easydict import EasyDict

cartpole_acer_config = dict(
    exp_name='cartpole_acer_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=1116,
            action_shape=26,
            encoder_hidden_size_list=[64, 64],
            critic_head_output=26,
        ),
        # (int) the trajectory length to calculate Q retrace target
        learn=dict(
            # (int) collect n_sample data, train model update_per_collect times
            # here we follow ppo serial pipeline
            update_per_collect=4,
            unroll_len=1,
            # (int) the number of data for a train iteration
            batch_size=16,
            learning_rate_actor=0.0005,
            learning_rate_critic=0.0005,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            # entropy_weight=0.0001,
            entropy_weight=0.0,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.9,
            # (float) additional discounting parameter
            # (int) the trajectory length to calculate v-trace target
            # (float) clip ratio of importance weights
            trust_region=True,
            c_clip_ratio=10,
            # (float) clip ratio of importance sampling
        ),
        # learn=dict(
        #     unroll_len=1,
        #     epoch_per_collect=2,
        #     batch_size=64,
        #     learning_rate=0.001,
        #     value_weight=0.5,
        #     entropy_weight=0.01,
        #     clip_ratio=0.2,
        #     learner=dict(hook=dict(save_ckpt_after_iter=100)),
        # ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=16,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.9,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=200, )),
        other=dict(replay_buffer=dict(replay_buffer_size=10000, ), ),
    ),
)

cartpole_acer_config = EasyDict(cartpole_acer_config)
main_config = cartpole_acer_config

cartpole_acer_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='acer'),
)

cartpole_acer_create_config = EasyDict(cartpole_acer_create_config)
create_config = cartpole_acer_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_acer_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
