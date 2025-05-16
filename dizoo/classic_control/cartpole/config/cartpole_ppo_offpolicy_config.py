from easydict import EasyDict
import torch.nn as nn

cartpole_ppo_offpolicy_config = dict(
    env_id=0,
    exp_name='wargame_ppo_offpolicy_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=2000,
    ),
    policy=dict(
        cuda=True,
        algorithm='ppo_offpolicy',
        model=dict(
            obs_shape=11,
            action_shape=8,
            encoder_hidden_size_list=[64, 128, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
            action_space='discrete',
            activation=nn.LeakyReLU(),
        ),
        learn=dict(
            update_per_collect=10,
            batch_size=2000,
            learning_rate=0.000001,
            value_weight=0.2,
            entropy_weight=0.01,
            clip_ratio=0.1,
            learner=dict(hook=dict(save_ckpt_after_iter=50)),
        ),
        collect=dict(
            n_sample=1000 * 4,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=20, )),
        other=dict(replay_buffer=dict(replay_buffer_size=50000))
    ),
)
cartpole_ppo_offpolicy_config = EasyDict(cartpole_ppo_offpolicy_config)
main_config = cartpole_ppo_offpolicy_config
cartpole_ppo_offpolicy_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
)
cartpole_ppo_offpolicy_create_config = EasyDict(cartpole_ppo_offpolicy_create_config)
create_config = cartpole_ppo_offpolicy_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_ppo_offpolicy_config.py -s 0`
    from ding.entry import serial_pipeline

    serial_pipeline((main_config, create_config), seed=0)
