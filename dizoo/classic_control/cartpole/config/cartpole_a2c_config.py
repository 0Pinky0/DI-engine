from easydict import EasyDict

cartpole_a2c_config = dict(
    env_id=0,
    exp_name='cartpole_a2c_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=2000,
    ),
    policy=dict(
        cuda=True,
        algorithm = 'a2c_10vs3',
        action_space='discrete',
        model=dict(
            obs_shape=11,
            action_shape=8,
            action_space='discrete',
            encoder_hidden_size_list=[64, 128, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        learn=dict(
            epoch_per_collect=50,
            batch_size=800,
            learning_rate=0.0025,
            value_weight=0.05, #0.5
            entropy_weight=0.01,
            clip_ratio=0.1,
            learner=dict(hook=dict(save_ckpt_after_iter=10)),
        ),
        collect=dict(
            n_sample=1000 * 1,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=3, ), ),
    ),
)
cartpole_a2c_config = EasyDict(cartpole_a2c_config)
main_config = cartpole_a2c_config

cartpole_a2c_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='a2c'),
)
cartpole_a2c_create_config = EasyDict(cartpole_a2c_create_config)
create_config = cartpole_a2c_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c cartpole_a2c_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
