from typing import TYPE_CHECKING, Callable, List, Tuple, Any
from functools import reduce
import treetensor.torch as ttorch
import numpy as np
from ditk import logging
from ding.utils import EasyTimer
from ding.envs import BaseEnvManager
from ding.policy import Policy
from ding.torch_utils import to_ndarray, get_shape0
import json
from easydict import EasyDict
import os

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext


class TransitionList:

    def __init__(self, env_num: int) -> None:
        self.env_num = env_num
        self._transitions = [[] for _ in range(env_num)]
        self._done_idx = [[] for _ in range(env_num)]

    def append(self, env_id: int, transition: Any) -> None:
        self._transitions[env_id].append(transition)
        if transition.done:
            self._done_idx[env_id].append(len(self._transitions[env_id]))

    def to_trajectories(self) -> Tuple[List[Any], List[int]]:
        trajectories = sum(self._transitions, [])
        lengths = [len(t) for t in self._transitions]
        trajectory_end_idx = [reduce(lambda x, y: x + y, lengths[:i + 1]) for i in range(len(lengths))]
        trajectory_end_idx = [t - 1 for t in trajectory_end_idx]
        return trajectories, trajectory_end_idx

    def to_episodes(self) -> List[List[Any]]:
        episodes = []
        for env_id in range(self.env_num):
            last_idx = 0
            for done_idx in self._done_idx[env_id]:
                episodes.append(self._transitions[env_id][last_idx:done_idx])
                last_idx = done_idx
        return episodes

    def clear(self):
        for item in self._transitions:
            item.clear()
        for item in self._done_idx:
            item.clear()


def inferencer(cfg: EasyDict, seed: int, policy: Policy, env: BaseEnvManager) -> Callable:
    """
    Overview:
        The middleware that executes the inference process.
    Arguments:
        - seed (:obj:`int`): Random seed.
        - policy (:obj:`Policy`): The policy to be inferred.
        - env (:obj:`BaseEnvManager`): The env where the inference process is performed. \
            The env.ready_obs (:obj:`tnp.array`) will be used as model input.
    """

    env.seed(seed)

    def _inference(ctx: "OnlineRLContext"):
        """
        Output of ctx:
            - obs (:obj:`Union[torch.Tensor, Dict[torch.Tensor]]`): The input observations collected \
                from all collector environments.
            - action: (:obj:`List[np.ndarray]`): The inferred actions listed by env_id.
            - inference_output (:obj:`Dict[int, Dict]`): The dict of which the key is env_id (int), \
                and the value is inference result (Dict).
        """

        if env.closed:
            env.launch()

        obs = ttorch.as_tensor(env.ready_obs)
        ctx.obs = obs
        obs = obs.to(dtype=ttorch.float32)
        # TODO mask necessary rollout

        obs = {i: obs[i] for i in range(get_shape0(obs))}  # TBD
        inference_output = policy.forward(obs, **ctx.collect_kwargs)
        if cfg.env_id == 1:
            ctx.action = [to_ndarray(v['action']) for v in inference_output.values()]
        else:
            # for i in range(len(np.array(inference_output[0]['logit']))):
                # max_value_index = np.argmax(
                # np.array(inference_output[0]['logit'][i, 2:]) ) + 2
                # inference_output[0]['action'][i] = max_value_index
            ctx.action = [to_ndarray(v['logit']) for v in inference_output.values()]
       
        ctx.inference_output = inference_output

    return _inference


def rolloutor(
        cfg: EasyDict,
        policy: Policy,
        env: BaseEnvManager,
        transitions: TransitionList,
        collect_print_freq=10,
) -> Callable:
    """
    Overview:
        The middleware that executes the transition process in the env.
    Arguments:
        - policy (:obj:`Policy`): The policy to be used during transition.
        - env (:obj:`BaseEnvManager`): The env for the collection, the BaseEnvManager object or \
                its derivatives are supported.
        - transitions (:obj:`TransitionList`): The transition information which will be filled \
            in this process, including `obs`, `next_obs`, `action`, `logit`, `value`, `reward` \
            and `done`.
    """
    env_episode_id = [_ for _ in range(env.env_num)]
    current_id = env.env_num
    timer = EasyTimer()
    last_train_iter = 0
    total_envstep_count = 0
    total_episode_count = 0
    total_train_sample_count = 0
    env_info = {env_id: {'time': 0., 'step': 0, 'train_sample': 0} for env_id in range(env.env_num)}
    episode_info = []

    def _rollout(ctx: "OnlineRLContext"):
        """
        Input of ctx:
            - action: (:obj:`List[np.ndarray]`): The inferred actions from previous inference process.
            - obs (:obj:`Dict[Tensor]`): The states fed into the transition dict.
            - inference_output (:obj:`Dict[int, Dict]`): The inference results to be fed into the \
                transition dict.
            - train_iter (:obj:`int`): The train iteration count to be fed into the transition dict.
            - env_step (:obj:`int`): The count of env step, which will increase by 1 for a single \
                transition call.
            - env_episode (:obj:`int`): The count of env episode, which will increase by 1 if the \
                trajectory stops.
        """

        nonlocal current_id, env_info, episode_info, timer, \
        total_episode_count, total_envstep_count, total_train_sample_count, last_train_iter
        timesteps = env.step(ctx.action)
        ctx.env_step += len(timesteps)
        timesteps = [t.tensor() for t in timesteps]

        collected_sample = 0
        collected_step = 0
        collected_episode = 0
        interaction_duration = timer.value / len(timesteps)
        for i, timestep in enumerate(timesteps):
            with timer:
                ctx.inference_output[i]['action']=timestep['info']['step_actions']
                transition = policy.process_transition(ctx.obs[i], ctx.inference_output[i], timestep)
                if 'logit' not in transition:
                    transition['logit'] = ctx.inference_output[i]['logit']
                # print(transition)
                transition = ttorch.as_tensor(transition)
                # print(transition)
                # quit()
                ops_transition = transition.clone()
                # print(transition)
                # 将[k, 1116]的张量拆成k个[1116]的张量
                k = transition['action'].shape[0]
                
                for i in range(k):
                    
                    if int(transition['action'][i]) != -1:
                        tensor_action = transition['action'][i]
                        tensor_action = tensor_action.unsqueeze(-1)
                        tensor_done = transition['done']
                        if 'next_obs' in transition:
                            tensor_next_obs = {}
                            tensor_next_obs['action_mask'] = transition['next_obs']['action_mask'][i]
                            try:
                                tensor_next_obs['agent_state'] = transition['next_obs']['agent_state'][i]
                            except IndexError:
                                print(len(transition['next_obs']['agent_state']))
                            ops_transition.next_obs = ttorch.as_tensor(tensor_next_obs)
                        if 'logit' in transition:
                            ops_transition['logit'] = transition['logit'][i]
                        if 'adv' in transition:
                            ops_transition['adv'] = transition['adv'][i]
                        tensor_obs = {}
                        tensor_obs['action_mask'] = transition['obs']['action_mask'][i]
                        tensor_obs['agent_state'] = transition['obs']['agent_state'][i]
                        tensor_reward = transition['reward'][i]
                        value = transition['value'][i]
                        ops_tran_new = {}
                        ops_tran_new['logit'] = ttorch.as_tensor(transition['logit'][i])
                        ops_tran_new['action'] = ttorch.as_tensor(tensor_action)
                        ops_tran_new['reward'] = ttorch.as_tensor(tensor_reward)
                        ops_tran_new['done'] = ttorch.as_tensor(tensor_done)
                        ops_tran_new['next_obs'] = ttorch.as_tensor(tensor_next_obs)
                        ops_tran_new['obs'] = ttorch.as_tensor(tensor_obs)
                        ops_tran_new['collect_train_iter'] = ttorch.as_tensor([ctx.train_iter])
                        ops_tran_new['env_data_id'] = ttorch.as_tensor([env_episode_id[timestep.env_id]])
                        ops_tran_new['value'] = ttorch.as_tensor(value)
                        ops_tran_new = ttorch.as_tensor(ops_tran_new)
                        transitions.append(timestep.env_id, ops_tran_new)
                        # ops_transition.action = tensor_action
                        # ops_transition.done = tensor_done
                        # ops_transition.obs = ttorch.as_tensor(tensor_obs)
                        # ops_transition.reward = tensor_reward
                        # ops_transition.collect_train_iter = ttorch.as_tensor([ctx.train_iter])
                        # ops_transition.env_data_id = ttorch.as_tensor([env_episode_id[timestep.env_id]])
                        # transitions.append(timestep.env_id, ops_transition)

                collected_step += 1
                collected_sample += len(ops_transition.obs['agent_state']) * k
                env_info[timestep.env_id.item()]['step'] += 1
                env_info[timestep.env_id.item()]['train_sample'] += len(ops_transition.obs['agent_state']) * k 
            

            env_info[timestep.env_id.item()]['time'] += timer.value + interaction_duration
            if timestep.done:
                info = {
                    # 'reward': timestep.reward,
                    'reward': timestep.info['eval_episode_return'],
                    'time': env_info[timestep.env_id.item()]['time'],
                    'step': env_info[timestep.env_id.item()]['step'],
                    'train_sample': env_info[timestep.env_id.item()]['train_sample'],
                }
                # print(collected_episode , " episode has done!!! ")
                episode_info.append(info)
                policy.reset([timestep.env_id.item()])
                env_episode_id[timestep.env_id.item()] = current_id
                collected_episode += 1
                current_id += 1
                ctx.env_episode += 1

        total_envstep_count += collected_step
        total_episode_count += collected_episode
        total_train_sample_count += collected_sample

        if (ctx.train_iter - last_train_iter) >= collect_print_freq and len(episode_info) > 0:
            output_log(episode_info, total_episode_count, total_envstep_count, total_train_sample_count, cfg)
            last_train_iter = ctx.train_iter

    return _rollout


def output_log(episode_info, total_episode_count, total_envstep_count, total_train_sample_count, cfg) -> None:
    """
    Overview:
        Print the output log information. You can refer to the docs of `Best Practice` to understand \
        the training generated logs and tensorboards.
    Arguments:
        - train_iter (:obj:`int`): the number of training iteration.
    """
    episode_count = len(episode_info)
    envstep_count = sum([d['step'] for d in episode_info])
    train_sample_count = sum([d['train_sample'] for d in episode_info])
    duration = sum([d['time'] for d in episode_info])
    episode_return = [d['reward'].item() for d in episode_info]
    info = {
        'episode_count': episode_count,
        'envstep_count': envstep_count,
        'train_sample_count': train_sample_count,
        'total_envstep_count': total_envstep_count,
        'total_train_sample_count': total_train_sample_count,
        'total_episode_count': total_episode_count,
        # 'each_reward': episode_return,
    }
    os.makedirs('logs/collect_json', exist_ok=True)

    # 檢查文件是否存在，如果不存在則創建
    file_path = f'logs/collect_json/{cfg.policy.algorithm}_log_data.json'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as json_file:
            json_file.write("")  # 創建一個空文件

    # 寫入日誌數據
    with open(file_path, 'a') as json_file:
        json_file.write(json.dumps(info, ensure_ascii=False) + ",\n")
    episode_info.clear()
    logging.info("collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()])))
