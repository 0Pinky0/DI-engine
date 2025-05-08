from typing import Callable, Any, List, Union, Optional
from abc import ABC, abstractmethod
from collections import deque
from ditk import logging
import numpy as np
import torch
import treetensor.numpy as tnp
import treetensor.torch as ttorch
from easydict import EasyDict
from ding.envs import BaseEnvManager
from ding.framework.context import Context, OfflineRLContext, OnlineRLContext
from ding.policy import Policy
from ding.data import Dataset, DataLoader
from ding.framework import task
from ding.torch_utils import to_ndarray, get_shape0
from ding.utils import lists_to_dicts
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import zipfile
import json
import time

class IMetric(ABC):

    @abstractmethod
    def eval(self, inputs: Any, label: Any) -> dict:
        raise NotImplementedError

    @abstractmethod
    def reduce_mean(self, inputs: List[Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def gt(self, metric1: Any, metric2: Any) -> bool:
        """
        Overview:
            Whether metric1 is greater than metric2 (>=)

        .. note::
            If metric2 is None, return True
        """
        raise NotImplementedError
    
class RewardMonitor:
    def __init__(self, cfg):
        self.rewards = []
        self.env_steps = []
        self.fig, self.ax = plt.subplots()
        self.ani = FuncAnimation(self.fig, self.update_plt, frames=self._generate_frames(cfg.policy.collect.n_sample), interval=1000)
        plt.show(block=False)  # 使用block=False允许交互式操作

    def _generate_frames(self, n_sample):
        # 动态生成帧数，根据实际需要调整
        return np.arange(0, n_sample, max(1, n_sample * 200))  # 确保至少有200个帧

    def update_plt(self, frame):
        # 假设episode_return和ctx是在外部调用时传入或通过其他方式更新的
        self.rewards.append(self.episode_return)  # 这里需要在外部调用时设置episode_return
        self.env_steps.append(self.env_step)  # 同样，ctx也需要适当设置

        self.ax.clear()  # 使用ax.clear()而不是plt.cla()以保持单个子图的上下文
        self.ax.plot(self.env_steps, self.rewards, 'r-', label='Reward Over Time')
        self.ax.set_title('Real-time Monitoring of Reward')
        self.ax.set_xlabel('Step')
        self.ax.set_ylabel('Reward')
        self.ax.legend()

    def update_data(self, episode_return, env_step):
        """外部调用此方法来更新数据"""
        self.episode_return = episode_return
        self.env_step = env_step


class VectorEvalMonitor(object):
    """
    Overview:
        In some cases,  different environment in evaluator may collect different length episode. For example, \
            suppose we want to collect 12 episodes in evaluator but only have 5 environments, if we didn’t do \
            any thing, it is likely that we will get more short episodes than long episodes. As a result, \
            our average reward will have a bias and may not be accurate. we use VectorEvalMonitor to solve the problem.
    Interfaces:
        __init__, is_finished, update_info, update_reward, get_episode_return, get_latest_reward, get_current_episode,\
            get_episode_info, update_video, get_episode_video
    """

    def __init__(self, env_num: int, n_episode: int) -> None:
        """
        Overview:
            Init method. According to the number of episodes and the number of environments, determine how many \
                episodes need to be opened for each environment, and initialize the reward, info and other \
                information
        Arguments:
            - env_num (:obj:`int`): the number of episodes need to be open
            - n_episode (:obj:`int`): the number of environments
        """
        assert n_episode >= env_num, "n_episode < env_num, please decrease the number of eval env"
        self._env_num = env_num
        self._n_episode = n_episode
        each_env_episode = [n_episode // env_num for _ in range(env_num)]
        for i in range(n_episode % env_num):
            each_env_episode[i] += 1
        self._reward = {env_id: deque(maxlen=maxlen) for env_id, maxlen in enumerate(each_env_episode)}
        self._info = {env_id: deque(maxlen=maxlen) for env_id, maxlen in enumerate(each_env_episode)}
        self._video = {
            env_id: deque([[] for _ in range(maxlen)], maxlen=maxlen)
            for env_id, maxlen in enumerate(each_env_episode)
        }
        self._output = {
            env_id: deque([[] for _ in range(maxlen)], maxlen=maxlen)
            for env_id, maxlen in enumerate(each_env_episode)
        }

    def is_finished(self) -> bool:
        """
        Overview:
            Determine whether the evaluator has completed the work.
        Return:
            - result: (:obj:`bool`): whether the evaluator has completed the work
        """
        return all([len(v) == v.maxlen for v in self._reward.values()])

    def update_info(self, env_id: int, info: Any) -> None:
        """
        Overview:
            Update the information of the environment indicated by env_id.
        Arguments:
            - env_id: (:obj:`int`): the id of the environment we need to update information
            - info: (:obj:`Any`): the information we need to update
        """
        self._info[env_id].append(info)

    def update_reward(self, env_id: Union[int, np.ndarray], reward: Any) -> None:
        """
        Overview:
            Update the reward indicated by env_id.
        Arguments:
            - env_id: (:obj:`int`): the id of the environment we need to update the reward
            - reward: (:obj:`Any`): the reward we need to update
        """
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        if isinstance(env_id, np.ndarray):
            env_id = env_id.item()
        self._reward[env_id].append(reward)

    def get_episode_return(self) -> list:
        """
        Overview:
            Sum up all reward and get the total return of one episode.
        """
        return sum([list(v) for v in self._reward.values()], [])  # sum(iterable, start)

    def get_latest_reward(self, env_id: int) -> int:
        """
        Overview:
            Get the latest reward of a certain environment.
        Arguments:
            - env_id: (:obj:`int`): the id of the environment we need to get reward.
        """
        return self._reward[env_id][-1]

    def get_current_episode(self) -> int:
        """
        Overview:
            Get the current episode. We can know which episode our evaluator is executing now.
        """
        return sum([len(v) for v in self._reward.values()])

    def get_episode_info(self) -> dict:
        """
        Overview:
            Get all episode information, such as total return of one episode.
        """
        if len(self._info[0]) == 0:
            return None
        else:
            # sum among all envs
            total_info = sum([list(v) for v in self._info.values()], [])
            if isinstance(total_info[0], tnp.ndarray):
                total_info = [t.json() for t in total_info]
            total_info = lists_to_dicts(total_info)
            new_dict = {}
            for k in total_info.keys():
                try:
                    if np.isscalar(total_info[k][0].item()):
                        new_dict[k + '_mean'] = np.mean(total_info[k])
                except:  # noqa
                    pass
            return new_dict

    def _select_idx(self):
        reward = [t.item() for t in self.get_episode_return()]
        sortarg = np.argsort(reward)
        # worst, median(s), best
        if len(sortarg) == 1:
            idxs = [sortarg[0]]
        elif len(sortarg) == 2:
            idxs = [sortarg[0], sortarg[-1]]
        elif len(sortarg) == 3:
            idxs = [sortarg[0], sortarg[len(sortarg) // 2], sortarg[-1]]
        else:
            # TensorboardX pad the number of videos to even numbers with black frames,
            # therefore providing even number of videos prevents black frames being rendered.
            idxs = [sortarg[0], sortarg[len(sortarg) // 2 - 1], sortarg[len(sortarg) // 2], sortarg[-1]]
        return idxs

    def update_video(self, imgs):
        for env_id, img in imgs.items():
            if len(self._reward[env_id]) == self._reward[env_id].maxlen:
                continue
            self._video[env_id][len(self._reward[env_id])].append(img)

    def get_episode_video(self):
        """
        Overview:
            Convert list of videos into [N, T, C, H, W] tensor, containing
            worst, median, best evaluation trajectories for video logging.
        """
        videos = sum([list(v) for v in self._video.values()], [])
        videos = [np.transpose(np.stack(video, 0), [0, 3, 1, 2]) for video in videos]
        idxs = self._select_idx()
        videos = [videos[idx] for idx in idxs]
        # pad videos to the same length with last frames
        max_length = max(video.shape[0] for video in videos)
        for i in range(len(videos)):
            if videos[i].shape[0] < max_length:
                padding = np.tile([videos[i][-1]], (max_length - videos[i].shape[0], 1, 1, 1))
                videos[i] = np.concatenate([videos[i], padding], 0)
        videos = np.stack(videos, 0)
        assert len(videos.shape) == 5, 'Need [N, T, C, H, W] input tensor for video logging!'
        return videos

    def update_output(self, output):
        for env_id, o in output.items():
            if len(self._reward[env_id]) == self._reward[env_id].maxlen:
                continue
            self._output[env_id][len(self._reward[env_id])].append(to_ndarray(o))

    def get_episode_output(self):
        output = sum([list(v) for v in self._output.values()], [])
        idxs = self._select_idx()
        output = [output[idx] for idx in idxs]
        return output


def interaction_evaluator(
        cfg: EasyDict, policy: Policy, env: BaseEnvManager, render: bool = False, **kwargs
) -> Callable:
    """
    Overview:
        The middleware that executes the evaluation.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - policy (:obj:`Policy`): The policy to be evaluated.
        - env (:obj:`BaseEnvManager`): The env for the evaluation.
        - render (:obj:`bool`): Whether to render env images and policy logits.
        - kwargs: (:obj:`Any`): Other arguments for specific evaluation.
    """
    if task.router.is_active and not task.has_role(task.role.EVALUATOR):
        return task.void()

    env.seed(cfg.seed, dynamic_seed=False)


    def save_replay(replay_name, data, cfg):
        zip_name = f"logs/replays/{cfg.policy.algorithm}/{replay_name}.zip"
        # if not os.path.exists("logs/replays/"):
            # os.makedirs("logs/replays/")
        dir_path = os.path.dirname(f"logs/replays/{cfg.policy.algorithm}/{replay_name}.zip")
        os.makedirs(dir_path, exist_ok=True)
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for i, ob in enumerate(data):
                d = json.dumps(ob, ensure_ascii=False, separators=(",", ":"))
                z.writestr(f"{i}", d)


    def _evaluate(ctx: Union["OnlineRLContext", "OfflineRLContext"]):
        """
        Overview:
            - The evaluation will be executed if the task begins and enough train_iter passed \
                since last evaluation.
        Input of ctx:
            - last_eval_iter (:obj:`int`): Last evaluation iteration.
            - train_iter (:obj:`int`): Current train iteration.
        Output of ctx:
            - eval_value (:obj:`float`): The average reward in the current evaluation.
        """

        # num += 1
        # evaluation will be executed if the task begins or enough train_iter after last evaluation
        if ctx.last_eval_iter != -1 and \
                (ctx.train_iter - ctx.last_eval_iter < cfg.policy.eval.evaluator.eval_freq):
            if ctx.train_iter != ctx.last_eval_iter:
                return
        if len(kwargs) > 0:
            kwargs_str = '/'.join([f'{k}({v})' for k, v in kwargs.items()])
        else:
            kwargs_str = ''

        if env.closed:
            env.launch()
        else:
            env.reset()
        policy.reset()
        eval_monitor = VectorEvalMonitor(env.env_num, cfg.env.n_evaluator_episode)
        # plt_monitor = RewardMonitor(cfg)

        while not eval_monitor.is_finished():
            obs = ttorch.as_tensor(env.ready_obs).to(dtype=ttorch.float32)
            obs = {i: obs[i] for i in range(get_shape0(obs))}  # TBD
            if len(kwargs) > 0:
                inference_output = policy.forward(obs, **kwargs)
            else:
                inference_output = policy.forward(obs)
            # print("inference_output:", inference_output)
            if render:
                eval_monitor.update_video(env.ready_imgs)
                eval_monitor.update_output(inference_output)
            output = [v for v in inference_output.values()]
            if cfg.env_id == 1:
                action= [to_ndarray(v['action']) for v in output]
            else:
                action = [to_ndarray(v['logit']) for v in output] 
            # print(action)
            timesteps = env.step(action)
            all_states = env.env_ref.all_states
            if len(all_states) > 1 :
                replay_states = all_states
            
            # print(len(all_states))               
            # print("timesteps", timesteps)
            for timestep in timesteps:
                env_id = timestep.env_id.item()
                if timestep.done:
                    reward = timestep.info.eval_episode_return
                    opt_reward = timestep.info.opt_return
                    game_id = f'{time.strftime("%Y-%m-%d-%H-%M-%S")}_{"policy"}_{cfg.policy.algorithm}_{"itr"}_{ctx.train_iter}_{"scout_test_target"}_{env.SelectPoint[0].target_pos}_{reward}'
                    # all_states = timestep.all_states
                    try:
                        save_replay(game_id, replay_states, cfg)
                    except NameError:
                        pass
                    policy.reset([env_id])
                    eval_monitor.update_reward(env_id, reward)
                    if 'episode_info' in timestep.info:
                        eval_monitor.update_info(env_id, timestep.info.episode_info)
                    # print("env.SelectPoint:",env.SelectPoint[0].target_pos)
                    # print("env.int_target_pos:", env.int_target_pos)
                    # print("env.env.director_ai.target_pos:", env.env[0].director_ai.target_pos)
        episode_return = eval_monitor.get_episode_return()
        episode_return_min = np.min(episode_return)
        episode_return_max = np.max(episode_return)
        episode_return_std = np.std(episode_return)
        episode_return = np.array(episode_return)
        # episode_return_filtered = episode_return[np.argsort(episode_return)[1:-1]]
        episode_return = np.mean(episode_return)
        stop_flag = episode_return >= cfg.env.stop_value and ctx.train_iter > 0
        # plt_monitor.update_data(episode_return, ctx.env_step)

        if isinstance(ctx, OnlineRLContext):
            logging.info(
                'Evaluation: Train Iter({}) Env Step({}) Episode Return({:.3f}) {} opt_return({})'.format(
                    ctx.train_iter, ctx.env_step, episode_return, kwargs_str, opt_reward
                )
            )
            # 将日志数据写入json保存reward
            log_data = {
                    "type": "Evaluation",
                    "train_iter": ctx.train_iter,
                    "env_step": ctx.env_step,
                    "opt_return": float(np.sum(opt_reward)),
                    "episode_return": float(episode_return),
                    "episode_return_max": float(episode_return_max),
                    "episode_return_std": float(episode_return_std),  # 确保浮点数格式

                    "additional_params": kwargs_str  # 假设kwargs_str可以直接作为字符串存入，或根据实际情况处理
                }
            import json
            os.makedirs('logs/evaluate_json', exist_ok=True)

            # 檢查文件是否存在，如果不存在則創建
            file_path = f'logs/evaluate_json/{cfg.policy.algorithm}_log_data_{cfg.time}.json'
            if not os.path.exists(file_path):
                with open(file_path, 'w') as json_file:
                    json_file.write("")  # 創建一個空文件

            # 寫入日誌數據
            with open(file_path, 'a') as json_file:
                json_file.write(json.dumps(log_data, ensure_ascii=False) + ",\n")
        elif isinstance(ctx, OfflineRLContext):
            logging.info(
                'Evaluation: Train Iter({}) Eval Return({:.3f}) {}'.format(ctx.train_iter, episode_return, kwargs_str)
            )
        else:
            raise TypeError("not supported ctx type: {}".format(type(ctx)))
        ctx.last_eval_iter = ctx.train_iter
        ctx.eval_value = episode_return
        ctx.eval_value_min = episode_return_min
        ctx.eval_value_max = episode_return_max
        ctx.eval_value_std = episode_return_std
        ctx.last_eval_value = ctx.eval_value
        ctx.eval_output = {'opt_return': float(np.sum(opt_reward))}
        episode_info = eval_monitor.get_episode_info()
        if episode_info is not None:
            ctx.eval_output['episode_info'] = episode_info
        if render:
            ctx.eval_output['replay_video'] = eval_monitor.get_episode_video()
            ctx.eval_output['output'] = eval_monitor.get_episode_output()
        else:
            ctx.eval_output['output'] = output  # for compatibility

        if len(kwargs) > 0:
            ctx.info_for_logging.update(
                {
                    f'{kwargs_str}/eval_episode_return': episode_return,
                    f'{kwargs_str}/eval_episode_return_min': episode_return_min,
                    f'{kwargs_str}/eval_episode_return_max': episode_return_max,
                    f'{kwargs_str}/eval_episode_return_std': episode_return_std,
                }
            )

        if stop_flag:
            task.finish = True



    return _evaluate



def interaction_evaluator_ttorch(
        seed: int,
        policy: Policy,
        env: BaseEnvManager,
        n_evaluator_episode: Optional[int] = None,
        stop_value: float = np.inf,
        eval_freq: int = 1000,
        render: bool = False,
) -> Callable:
    """
    Overview:
        The middleware that executes the evaluation with ttorch data.
    Arguments:
        - policy (:obj:`Policy`): The policy to be evaluated.
        - env (:obj:`BaseEnvManager`): The env for the evaluation.
        - render (:obj:`bool`): Whether to render env images and policy logits.
    """
    if task.router.is_active and not task.has_role(task.role.EVALUATOR):
        return task.void()

    env.seed(seed, dynamic_seed=False)
    if n_evaluator_episode is None:
        n_evaluator_episode = env.env_num

    def _evaluate(ctx: "OnlineRLContext"):
        """
        Overview:
            - The evaluation will be executed if the task begins and enough train_iter passed \
                since last evaluation.
        Input of ctx:
            - last_eval_iter (:obj:`int`): Last evaluation iteration.
            - train_iter (:obj:`int`): Current train iteration.
        Output of ctx:
            - eval_value (:obj:`float`): The average reward in the current evaluation.
        """

        # evaluation will be executed if the task begins or enough train_iter after last evaluation
        if ctx.last_eval_iter != -1 and (ctx.train_iter - ctx.last_eval_iter < eval_freq):
            return

        if env.closed:
            env.launch()
        else:
            env.reset()
        policy.reset()
        device = policy._device
        eval_monitor = VectorEvalMonitor(env.env_num, n_evaluator_episode)

        while not eval_monitor.is_finished():
            obs = ttorch.as_tensor(env.ready_obs).to(dtype=ttorch.float32)
            obs = obs.to(device)
            inference_output = policy.eval(obs)
            inference_output = inference_output.cpu()
            if render:
                eval_monitor.update_video(env.ready_imgs)
                # eval_monitor.update_output(inference_output)
            action = inference_output.action.numpy()
            timesteps = env.step(action)
            for timestep in timesteps:
                env_id = timestep.env_id.item()
                if timestep.done:
                    policy.reset([env_id])
                    reward = timestep.info.eval_episode_return
                    eval_monitor.update_reward(env_id, reward)
                    if 'episode_info' in timestep.info:
                        eval_monitor.update_info(env_id, timestep.info.episode_info)
        episode_return = eval_monitor.get_episode_return()
        episode_return_std = np.std(episode_return)
        episode_return_mean = np.mean(episode_return)
        stop_flag = episode_return_mean >= stop_value and ctx.train_iter > 0
        logging.info(
            'Evaluation: Train Iter({})\tEnv Step({})\tMean Episode Return({:.3f})'.format(
                ctx.train_iter, ctx.env_step, episode_return_mean
            )
        )
        ctx.last_eval_iter = ctx.train_iter
        ctx.eval_value = episode_return_mean
        ctx.eval_value_std = episode_return_std
        ctx.last_eval_value = ctx.eval_value
        ctx.eval_output = {'episode_return': episode_return}
        episode_info = eval_monitor.get_episode_info()
        if episode_info is not None:
            ctx.eval_output['episode_info'] = episode_info
        if render:
            ctx.eval_output['replay_video'] = eval_monitor.get_episode_video()
            ctx.eval_output['output'] = eval_monitor.get_episode_output()
        else:
            ctx.eval_output['output'] = inference_output.numpy()  # for compatibility

        if stop_flag:
            task.finish = True

    return _evaluate


def metric_evaluator(cfg: EasyDict, policy: Policy, dataset: Dataset, metric: IMetric) -> Callable:
    dataloader = DataLoader(dataset, batch_size=cfg.policy.eval.batch_size)

    def _evaluate(ctx: "Context"):
        # evaluation will be executed if the task begins or enough train_iter after last evaluation
        if ctx.last_eval_iter != -1 and \
           (ctx.train_iter - ctx.last_eval_iter < cfg.policy.eval.evaluator.eval_freq):
            return

        policy.reset()
        eval_output = []

        for batch_idx, batch_data in enumerate(dataloader):
            inputs, label = batch_data
            inference_output = policy.forward(inputs)
            eval_output.append(metric.eval(inference_output, label))
        # TODO reduce avg_eval_output among different gpus
        avg_eval_output = metric.reduce_mean(eval_output)
        stop_flag = metric.gt(avg_eval_output, cfg.env.stop_value) and ctx.train_iter > 0
        logging.info(
            'Evaluation: Train Iter({})\tEnv Step({})\tEpisode Return({:.3f})'.format(
                ctx.train_iter, ctx.env_step, avg_eval_output
            )
        )
        ctx.last_eval_iter = ctx.train_iter
        ctx.eval_value = avg_eval_output

        if stop_flag:
            task.finish = True

    return _evaluate


# TODO battle evaluator
