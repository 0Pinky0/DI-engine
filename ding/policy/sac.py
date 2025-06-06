from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ding.torch_utils import Adam, to_device
from ding.rl_utils import (v_1step_td_data, v_1step_td_error, get_train_sample, q_v_1step_td_error, q_v_1step_td_data,
                           q_nstep_td_data, q_nstep_td_error, get_nstep_return_data)
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('discrete_sac')
class DiscreteSACPolicy(Policy):
    """
    Overview:
        Policy class of discrete SAC algorithm. Paper link: https://arxiv.org/abs/1910.07207.
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='discrete_sac',
        # (bool) Whether to use cuda for network and loss computation.
        cuda=False,
        # (bool) Whether to belong to on-policy or off-policy algorithm, DiscreteSAC is an off-policy algorithm.
        on_policy=False,
        # (bool) Whether to use priority sampling in buffer. Default to False in DiscreteSAC.
        priority=False,
        # (bool) Whether use Importance Sampling weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples (randomly collected) in replay buffer when training starts.
        random_collect_size=10000,
        # (bool) Whether to need policy-specific data in process transition.
        transition_with_policy_data=True,
        # (bool) Whether to enable multi-agent training setting.
        multi_agent=False,
        model=dict(
            # (bool) Whether to use double-soft-q-net for target q computation.
            # For more details, please refer to TD3 about Clipped Double-Q Learning trick.
            twin_critic=True,
        ),
        # learn_mode config
        learn=dict(
            # (int) How many updates (iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            update_per_collect=1,
            # (int) Minibatch size for one gradient descent.
            batch_size=256,
            # (float) Learning rate for soft q network.
            learning_rate_q=3e-4,
            # (float) Learning rate for policy network.
            learning_rate_policy=3e-4,
            # (float) Learning rate for auto temperature parameter `\alpha`.
            learning_rate_alpha=3e-4,
            # (float) Used for soft update of the target network,
            # aka. Interpolation factor in EMA update for target network.
            target_theta=0.005,
            # (float) Discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            # (float) Entropy regularization coefficient in SAC.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # If auto_alpha is set  to `True`, alpha is initialization for auto `\alpha`.
            alpha=0.2,
            # (bool) Whether to use auto temperature parameter `\alpha` .
            # Temperature parameter `\alpha` determines the relative importance of the entropy term against the reward.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # Note that: Using auto alpha needs to set the above `learning_rate_alpha`.
            auto_alpha=True,
            # (bool) Whether to use auto `\alpha` in log space.
            log_space=True,
            # (float) Target policy entropy value for auto temperature (alpha) adjustment.
            target_entropy=None,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float) Weight uniform initialization max range in the last output layer
            init_w=3e-3,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] shoule be set.
            n_sample=1,
            # (int) Split episodes or trajectories into pieces with length `unroll_len`.
            unroll_len=1,
            # (bool) Whether to collect logit in `process_transition`.
            # In some algorithm like guided cost learning, we need to use logit to train the reward model.
            collector_logit=False,
        ),
        eval=dict(),  # for compability
        other=dict(
            replay_buffer=dict(
                # (int) Maximum size of replay buffer. Usually, larger buffer size is good
                # for SAC but cost more storage.
                replay_buffer_size=1000000,
            ),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.
        """
        if self._cfg.multi_agent:
            return 'discrete_maqac', ['ding.model.template.maqac']
        else:
            return 'discrete_qac', ['ding.model.template.qac']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For DiscreteSAC, it mainly \
            contains three optimizers, algorithm-specific arguments such as gamma and twin_critic, main and target \
            model. Especially, the ``auto_alpha`` mechanism for balancing max entropy target is also initialized here.
            This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._twin_critic = self._cfg.model.twin_critic

        self._optimizer_q = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_q,
        )
        self._optimizer_policy = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_policy,
        )

        # Algorithm-Specific Config
        self._gamma = self._cfg.learn.discount_factor
        if self._cfg.learn.auto_alpha:
            if self._cfg.learn.target_entropy is None:
                assert 'action_shape' in self._cfg.model, "DiscreteSAC need network model with action_shape variable"
                self._target_entropy = -np.prod(self._cfg.model.action_shape)
            else:
                self._target_entropy = self._cfg.learn.target_entropy
            if self._cfg.learn.log_space:
                self._log_alpha = torch.log(torch.FloatTensor([self._cfg.learn.alpha]))
                self._log_alpha = self._log_alpha.to(self._device).requires_grad_()
                self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=self._cfg.learn.learning_rate_alpha)
                assert self._log_alpha.shape == torch.Size([1]) and self._log_alpha.requires_grad
                self._alpha = self._log_alpha.detach().exp()
                self._auto_alpha = True
                self._log_space = True
            else:
                self._alpha = torch.FloatTensor([self._cfg.learn.alpha]).to(self._device).requires_grad_()
                self._alpha_optim = torch.optim.Adam([self._alpha], lr=self._cfg.learn.learning_rate_alpha)
                self._auto_alpha = True
                self._log_space = False
        else:
            self._alpha = torch.tensor(
                [self._cfg.learn.alpha], requires_grad=False, device=self._device, dtype=torch.float32
            )
            self._auto_alpha = False

        # Main and target models
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, action, priority.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For SAC, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``logit``, ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys like ``weight``.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The information dict that indicated training result, which will be \
                recorded in text log and tensorboard, values must be python scalar or a list of scalars. For the \
                detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for DiscreteSACPolicy: \
            ``ding.policy.tests.test_discrete_sac``.
        """
        loss_dict = {}
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        obs = data['obs']
        next_obs = data['next_obs']
        reward = data['reward']
        done = data['done']
        logit = data['logit']
        action = data['action']

        # 1. predict q value
        q_value = self._learn_model.forward(obs, mode='compute_critic')['q_value']
        dist = torch.distributions.categorical.Categorical(logits=logit)
        dist_entropy = dist.entropy()
        entropy = dist_entropy.mean()

        # 2. predict target value

        # target q value. SARSA: first predict next action, then calculate next q value
        with torch.no_grad():
            policy_output_next = self._learn_model.forward(next_obs, mode='compute_actor')
            if self._cfg.multi_agent:
                policy_output_next['logit'][policy_output_next['action_mask'] == 0.0] = -1e8
            prob = F.softmax(policy_output_next['logit'], dim=-1)
            log_prob = torch.log(prob + 1e-8)
            target_q_value = self._target_model.forward(next_obs, mode='compute_critic')['q_value']
            # the value of a policy according to the maximum entropy objective
            if self._twin_critic:
                # find min one as target q value
                target_value = (
                    prob * (torch.min(target_q_value[0], target_q_value[1]) - self._alpha * log_prob.squeeze(-1))
                ).sum(dim=-1)
            else:
                target_value = (prob * (target_q_value - self._alpha * log_prob.squeeze(-1))).sum(dim=-1)

        # 3. compute q loss
        if self._twin_critic:
            q_data0 = q_v_1step_td_data(q_value[0].float(), target_value.float(), action.float(), reward.float(), done.float(), data['weight'])
            loss_dict['critic_loss'], td_error_per_sample0 = q_v_1step_td_error(q_data0, self._gamma)
            q_data1 = q_v_1step_td_data(q_value[1].float(), target_value.float(), action.float(), reward.float(), done.float(), data['weight'])
            loss_dict['twin_critic_loss'], td_error_per_sample1 = q_v_1step_td_error(q_data1, self._gamma)
            td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
        else:
            q_data = q_v_1step_td_data(q_value.float(), target_value.float(), action.float(), reward.float(), done.float(), data['weight'])
            loss_dict['critic_loss'], td_error_per_sample = q_v_1step_td_error(q_data, self._gamma)

        # 4. update q network
        self._optimizer_q.zero_grad()
        loss_dict['critic_loss'].backward()
        if self._twin_critic:
            loss_dict['twin_critic_loss'] = loss_dict['twin_critic_loss'].to(torch.float)
            loss_dict['twin_critic_loss'].backward()
        self._optimizer_q.step()

        # 5. evaluate to get action distribution
        policy_output = self._learn_model.forward(obs, mode='compute_actor')
        # 6. apply discrete action mask in multi_agent setting
        if self._cfg.multi_agent:
            policy_output['logit'][policy_output['action_mask'] == 0.0] = -1e8
        logit = policy_output['logit']
        prob = F.softmax(logit, dim=-1)
        log_prob = F.log_softmax(logit, dim=-1)

        with torch.no_grad():
            new_q_value = self._learn_model.forward(obs, mode='compute_critic')['q_value']
            if self._twin_critic:
                new_q_value = torch.min(new_q_value[0], new_q_value[1])
        # 7. compute policy loss
        # we need to sum different actions' policy loss and calculate the average value of a batch
        policy_loss = (prob * (self._alpha * log_prob - new_q_value)).sum(dim=-1).mean()

        loss_dict['policy_loss'] = policy_loss

        # 8. update policy network
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()

        # 9. compute alpha loss
        if self._auto_alpha:
            if self._log_space:
                log_prob = log_prob + self._target_entropy
                loss_dict['alpha_loss'] = (-prob.detach() * (self._log_alpha * log_prob.detach())).sum(dim=-1).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha = self._log_alpha.detach().exp()
            else:
                log_prob = log_prob + self._target_entropy
                loss_dict['alpha_loss'] = (-prob.detach() * (self._alpha * log_prob.detach())).sum(dim=-1).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha.data = torch.where(self._alpha > 0, self._alpha,
                                               torch.zeros_like(self._alpha)).requires_grad_()
        loss_dict['total_loss'] = sum(loss_dict.values())

        # target update
        self._target_model.update(self._learn_model.state_dict())
        return {
            'total_loss': loss_dict['total_loss'].item(),
            'policy_loss': loss_dict['policy_loss'].item(),
            'critic_loss': loss_dict['critic_loss'].item(),
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.detach().mean().item(),
            'alpha': self._alpha.item(),
            'q_value_1': target_q_value[0].detach().mean().item(),
            'q_value_2': target_q_value[1].detach().mean().item(),
            'target_value': target_value.detach().mean().item(),
            'entropy': entropy.item(),
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model, target_model and optimizers.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        ret = {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_q': self._optimizer_q.state_dict(),
            'optimizer_policy': self._optimizer_policy.state_dict(),
        }
        if self._auto_alpha:
            ret.update({'optimizer_alpha': self._alpha_optim.state_dict()})
        return ret

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer_q.load_state_dict(state_dict['optimizer_q'])
        self._optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        if self._auto_alpha:
            self._alpha_optim.load_state_dict(state_dict['optimizer_alpha'])

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For SAC, it contains the \
            collect_model to balance the exploration and exploitation with the epsilon and multinomial sample \
            mechanism, and other algorithm-specific arguments such as unroll_len. \
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        # Empirically, we found that eps_greedy_multinomial_sample works better than multinomial_sample
        # and eps_greedy_sample, and we don't divide logit by alpha,
        # for the details please refer to ding/model/wrapper/model_wrappers
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_multinomial_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of collect mode (collecting training data by interacting with envs). Forward means \
            that the policy gets some necessary data (mainly observation) from the envs and then returns the output \
            data, such as the action to interact with the envs. Besides, this policy also needs ``eps`` argument for \
            exploration, i.e., classic epsilon-greedy exploration strategy.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
            - eps (:obj:`float`): The epsilon value for exploration.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action and \
                other necessary data for learn mode defined in ``self._process_transition`` method. The key of the \
                dict is the same as the input data, i.e. environment id.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for DiscreteSACPolicy: \
            ``ding.policy.tests.test_discrete_sac``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor', eps=1.0)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: torch.Tensor, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For discrete SAC, it contains obs, next_obs, logit, action, reward, done.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For discrete SAC, it contains the action and the logit of the action.
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': policy_output['action'],
            'logit': policy_output['logit'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. In discrete SAC, a train sample is a processed transition (unroll_len=1).
        Arguments:
            - transitions (:obj:`List[Dict[str, Any]`): The trajectory data (a list of transition), each element is \
                the same format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each element is the similar format \
                as input transitions, but may contain more data for training.
        """
        return get_train_sample(transitions, self._unroll_len)

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For DiscreteSAC, it contains \
            the eval model to greedily select action type with argmax q_value mechanism.
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action. The \
                key of the dict is the same as the input data, i.e. environment id.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for DiscreteSACPolicy: \
            ``ding.policy.tests.test_discrete_sac``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        twin_critic = ['twin_critic_loss'] if self._twin_critic else []
        if self._auto_alpha:
            return super()._monitor_vars_learn() + [
                'alpha_loss', 'policy_loss', 'critic_loss', 'cur_lr_q', 'cur_lr_p', 'target_q_value', 'q_value_1',
                'q_value_2', 'alpha', 'td_error', 'target_value', 'entropy'
            ] + twin_critic
        else:
            return super()._monitor_vars_learn() + [
                'policy_loss', 'critic_loss', 'cur_lr_q', 'cur_lr_p', 'target_q_value', 'q_value_1', 'q_value_2',
                'alpha', 'td_error', 'target_value', 'entropy'
            ] + twin_critic


@POLICY_REGISTRY.register('sac')
class SACPolicy(Policy):
    """
    Overview:
        Policy class of continuous SAC algorithm. Paper link: https://arxiv.org/pdf/1801.01290.pdf

    Config:
           == ====================  ========    =============  ================================= =======================
           ID Symbol                Type        Default Value  Description                       Other
           == ====================  ========    =============  ================================= =======================
           1  ``type``              str         sac            | RL policy register name, refer  | this arg is optional,
                                                               | to registry ``POLICY_REGISTRY`` | a placeholder
           2  ``cuda``              bool        True           | Whether to use cuda for network |
           3  ``on_policy``         bool        False          | SAC is an off-policy            |
                                                               | algorithm.                      |
           4  ``priority``          bool        False          | Whether to use priority         |
                                                               | sampling in buffer.             |
           5  | ``priority_IS_``    bool        False          | Whether use Importance Sampling |
              | ``weight``                                     | weight to correct biased update |
           6  | ``random_``         int         10000          | Number of randomly collected    | Default to 10000 for
              | ``collect_size``                               | training samples in replay      | SAC, 25000 for DDPG/
              |                                                | buffer when training starts.    | TD3.
           7  | ``learn.learning``  float       3e-4           | Learning rate for soft q        | Defalut to 1e-3
              | ``_rate_q``                                    | network.                        |
           8  | ``learn.learning``  float       3e-4           | Learning rate for policy        | Defalut to 1e-3
              | ``_rate_policy``                               | network.                        |
           9  | ``learn.alpha``     float       0.2            | Entropy regularization          | alpha is initiali-
              |                                                | coefficient.                    | zation for auto
              |                                                |                                 | alpha, when
              |                                                |                                 | auto_alpha is True
           10 | ``learn.``          bool        False          | Determine whether to use        | Temperature parameter
              | ``auto_alpha``                                 | auto temperature parameter      | determines the
              |                                                | alpha.                          | relative importance
              |                                                |                                 | of the entropy term
              |                                                |                                 | against the reward.
           11 | ``learn.-``         bool        False          | Determine whether to ignore     | Use ignore_done only
              | ``ignore_done``                                | done flag.                      | in env like Pendulum
           12 | ``learn.-``         float       0.005          | Used for soft update of the     | aka. Interpolation
              | ``target_theta``                               | target network.                 | factor in polyak aver
              |                                                |                                 | aging for target
              |                                                |                                 | networks.
           == ====================  ========    =============  ================================= =======================
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='sac',
        # (bool) Whether to use cuda for network and loss computation.
        cuda=False,
        # (bool) Whether to belong to on-policy or off-policy algorithm, SAC is an off-policy algorithm.
        on_policy=False,
        # (bool) Whether to use priority sampling in buffer. Default to False in SAC.
        priority=False,
        # (bool) Whether use Importance Sampling weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples (randomly collected) in replay buffer when training starts.
        random_collect_size=10000,
        # (bool) Whether to need policy-specific data in process transition.
        transition_with_policy_data=True,
        # (bool) Whether to enable multi-agent training setting.
        multi_agent=False,
        model=dict(
            # (bool) Whether to use double-soft-q-net for target q computation.
            # For more details, please refer to TD3 about Clipped Double-Q Learning trick.
            twin_critic=True,
            # (str) Use reparameterization trick for continous action.
            action_space='reparameterization',
        ),
        # learn_mode config
        learn=dict(
            # (int) How many updates (iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            update_per_collect=1,
            # (int) Minibatch size for one gradient descent.
            batch_size=256,
            # (float) Learning rate for soft q network.
            learning_rate_q=3e-4,
            # (float) Learning rate for policy network.
            learning_rate_policy=3e-4,
            # (float) Learning rate for auto temperature parameter `\alpha`.
            learning_rate_alpha=3e-4,
            # (float) Used for soft update of the target network,
            # aka. Interpolation factor in EMA update for target network.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            # (float) Entropy regularization coefficient in SAC.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # If auto_alpha is set  to `True`, alpha is initialization for auto `\alpha`.
            alpha=0.2,
            # (bool) Whether to use auto temperature parameter `\alpha` .
            # Temperature parameter `\alpha` determines the relative importance of the entropy term against the reward.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # Note that: Using auto alpha needs to set the above `learning_rate_alpha`.
            auto_alpha=True,
            # (bool) Whether to use auto `\alpha` in log space.
            log_space=True,
            # (float) Target policy entropy value for auto temperature (alpha) adjustment.
            target_entropy=None,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float) Weight uniform initialization max range in the last output layer.
            init_w=3e-3,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            n_sample=1,
            # (int) Split episodes or trajectories into pieces with length `unroll_len`.
            unroll_len=1,
            # (bool) Whether to collect logit in `process_transition`.
            # In some algorithm like guided cost learning, we need to use logit to train the reward model.
            collector_logit=False,
        ),
        eval=dict(),  # for compability
        other=dict(
            replay_buffer=dict(
                # (int) Maximum size of replay buffer. Usually, larger buffer size is good
                # for SAC but cost more storage.
                replay_buffer_size=1000000,
            ),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.
        """
        if self._cfg.multi_agent:
            return 'continuous_maqac', ['ding.model.template.maqac']
        else:
            return 'continuous_qac', ['ding.model.template.qac']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For SAC, it mainly \
            contains three optimizers, algorithm-specific arguments such as gamma and twin_critic, main and target \
            model. Especially, the ``auto_alpha`` mechanism for balancing max entropy target is also initialized here.
            This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._twin_critic = self._cfg.model.twin_critic

        # Weight Init for the last output layer
        if hasattr(self._model, 'actor_head'):  # keep compatibility
            init_w = self._cfg.learn.init_w
            self._model.actor_head[-1].mu.weight.data.uniform_(-init_w, init_w)
            self._model.actor_head[-1].mu.bias.data.uniform_(-init_w, init_w)
            self._model.actor_head[-1].log_sigma_layer.weight.data.uniform_(-init_w, init_w)
            self._model.actor_head[-1].log_sigma_layer.bias.data.uniform_(-init_w, init_w)

        self._optimizer_q = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_q,
        )
        self._optimizer_policy = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_policy,
        )

        # Algorithm-Specific Config
        self._gamma = self._cfg.learn.discount_factor
        if self._cfg.learn.auto_alpha:
            if self._cfg.learn.target_entropy is None:
                assert 'action_shape' in self._cfg.model, "SAC need network model with action_shape variable"
                self._target_entropy = -np.prod(self._cfg.model.action_shape)
            else:
                self._target_entropy = self._cfg.learn.target_entropy
            if self._cfg.learn.log_space:
                self._log_alpha = torch.log(torch.FloatTensor([self._cfg.learn.alpha]))
                self._log_alpha = self._log_alpha.to(self._device).requires_grad_()
                self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=self._cfg.learn.learning_rate_alpha)
                assert self._log_alpha.shape == torch.Size([1]) and self._log_alpha.requires_grad
                self._alpha = self._log_alpha.detach().exp()
                self._auto_alpha = True
                self._log_space = True
            else:
                self._alpha = torch.FloatTensor([self._cfg.learn.alpha]).to(self._device).requires_grad_()
                self._alpha_optim = torch.optim.Adam([self._alpha], lr=self._cfg.learn.learning_rate_alpha)
                self._auto_alpha = True
                self._log_space = False
        else:
            self._alpha = torch.tensor(
                [self._cfg.learn.alpha], requires_grad=False, device=self._device, dtype=torch.float32
            )
            self._auto_alpha = False

        # Main and target models
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, action, priority.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For SAC, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys such as ``weight``.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The information dict that indicated training result, which will be \
                recorded in text log and tensorboard, values must be python scalar or a list of scalars. For the \
                detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for SACPolicy: ``ding.policy.tests.test_sac``.
        """
        loss_dict = {}
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        obs = data['obs']
        next_obs = data['next_obs']
        reward = data['reward']
        done = data['done']

        # 1. predict q value
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']

        # 2. predict target value
        with torch.no_grad():
            (mu, sigma) = self._learn_model.forward(next_obs, mode='compute_actor')['logit']

            dist = Independent(Normal(mu, sigma), 1)
            pred = dist.rsample()
            next_action = torch.tanh(pred)
            y = 1 - next_action.pow(2) + 1e-6
            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            next_log_prob = dist.log_prob(pred).unsqueeze(-1)
            next_log_prob = next_log_prob - torch.log(y).sum(-1, keepdim=True)

            next_data = {'obs': next_obs, 'action': next_action}
            target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
            # the value of a policy according to the maximum entropy objective
            if self._twin_critic:
                # find min one as target q value
                target_q_value = torch.min(target_q_value[0],
                                           target_q_value[1]) - self._alpha * next_log_prob.squeeze(-1)
            else:
                target_q_value = target_q_value - self._alpha * next_log_prob.squeeze(-1)

        # 3. compute q loss
        if self._twin_critic:
            q_data0 = v_1step_td_data(q_value[0], target_q_value, reward, done, data['weight'])
            loss_dict['critic_loss'], td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
            q_data1 = v_1step_td_data(q_value[1], target_q_value, reward, done, data['weight'])
            loss_dict['twin_critic_loss'], td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
            td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
        else:
            q_data = v_1step_td_data(q_value, target_q_value, reward, done, data['weight'])
            loss_dict['critic_loss'], td_error_per_sample = v_1step_td_error(q_data, self._gamma)

        # 4. update q network
        self._optimizer_q.zero_grad()
        if self._twin_critic:
            (loss_dict['critic_loss'] + loss_dict['twin_critic_loss']).backward()
        else:
            loss_dict['critic_loss'].backward()
        self._optimizer_q.step()

        # 5. evaluate to get action distribution
        (mu, sigma) = self._learn_model.forward(data['obs'], mode='compute_actor')['logit']
        dist = Independent(Normal(mu, sigma), 1)
        pred = dist.rsample()
        action = torch.tanh(pred)
        y = 1 - action.pow(2) + 1e-6
        # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
        log_prob = dist.log_prob(pred).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)

        eval_data = {'obs': obs, 'action': action}
        new_q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
        if self._twin_critic:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])

        # 6. compute policy loss
        policy_loss = (self._alpha * log_prob - new_q_value.unsqueeze(-1)).mean()

        loss_dict['policy_loss'] = policy_loss

        # 7. update policy network
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()

        # 8. compute alpha loss
        if self._auto_alpha:
            if self._log_space:
                log_prob = log_prob + self._target_entropy
                loss_dict['alpha_loss'] = -(self._log_alpha * log_prob.detach()).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha = self._log_alpha.detach().exp()
            else:
                log_prob = log_prob + self._target_entropy
                loss_dict['alpha_loss'] = -(self._alpha * log_prob.detach()).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha = max(0, self._alpha)

        loss_dict['total_loss'] = sum(loss_dict.values())

        # target update
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.detach().mean().item(),
            'alpha': self._alpha.item(),
            'target_q_value': target_q_value.detach().mean().item(),
            'transformed_log_prob': log_prob.mean().item(),
            **loss_dict
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model, target_model and optimizers.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        ret = {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_q': self._optimizer_q.state_dict(),
            'optimizer_policy': self._optimizer_policy.state_dict(),
        }
        if self._auto_alpha:
            ret.update({'optimizer_alpha': self._alpha_optim.state_dict()})
        return ret

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer_q.load_state_dict(state_dict['optimizer_q'])
        self._optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        if self._auto_alpha:
            self._alpha_optim.load_state_dict(state_dict['optimizer_alpha'])

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For SAC, it contains the \
            collect_model other algorithm-specific arguments such as unroll_len. \
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='base')
        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any], **kwargs) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of collect mode (collecting training data by interacting with envs). Forward means \
            that the policy gets some necessary data (mainly observation) from the envs and then returns the output \
            data, such as the action to interact with the envs.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action and \
                other necessary data for learn mode defined in ``self._process_transition`` method. The key of the \
                dict is the same as the input data, i.e. environment id.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            ``logit`` in SAC means the mu and sigma of Gaussioan distribution. Here we use this name for consistency.

        .. note::
            For more detailed examples, please refer to our unittest for SACPolicy: ``ding.policy.tests.test_sac``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            (mu, sigma) = self._collect_model.forward(data, mode='compute_actor')['logit']
            dist = Independent(Normal(mu, sigma), 1)
            action = torch.tanh(dist.rsample())
            output = {'logit': (mu, sigma), 'action': action}
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: torch.Tensor, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For continuous SAC, it contains obs, next_obs, action, reward, done. The logit \
            will be also added when ``collector_logit`` is True.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For continuous SAC, it contains the action and the logit (mu and sigma) of the action.
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.
        """
        if self._cfg.collect.collector_logit:
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'logit': policy_output['logit'],
                'action': policy_output['action'],
                'reward': timestep.reward,
                'done': timestep.done,
            }
        else:
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'action': policy_output['action'],
                'reward': timestep.reward,
                'done': timestep.done,
            }
        return transition

    def _get_train_sample(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. In continuous SAC, a train sample is a processed transition \
            (unroll_len=1).
        Arguments:
            - transitions (:obj:`List[Dict[str, Any]`): The trajectory data (a list of transition), each element is \
                the same format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each element is the similar format \
                as input transitions, but may contain more data for training.
        """
        return get_train_sample(transitions, self._unroll_len)

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For SAC, it contains the \
            eval model, which is equipped with ``base`` model wrapper to ensure compability.
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action. The \
                key of the dict is the same as the input data, i.e. environment id.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            ``logit`` in SAC means the mu and sigma of Gaussioan distribution. Here we use this name for consistency.

        .. note::
            For more detailed examples, please refer to our unittest for SACPolicy: ``ding.policy.tests.test_sac``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            (mu, sigma) = self._eval_model.forward(data, mode='compute_actor')['logit']
            action = torch.tanh(mu)  # deterministic_eval
            output = {'action': action}
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        twin_critic = ['twin_critic_loss'] if self._twin_critic else []
        alpha_loss = ['alpha_loss'] if self._auto_alpha else []
        return [
            'value_loss'
            'alpha_loss',
            'policy_loss',
            'critic_loss',
            'cur_lr_q',
            'cur_lr_p',
            'target_q_value',
            'alpha',
            'td_error',
            'transformed_log_prob',
        ] + twin_critic + alpha_loss


@POLICY_REGISTRY.register('sqil_sac')
class SQILSACPolicy(SACPolicy):
    """
    Overview:
        Policy class of continuous SAC algorithm with SQIL extension.
        SAC paper link: https://arxiv.org/pdf/1801.01290.pdf
        SQIL paper link: https://arxiv.org/abs/1905.11108
    """

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For SAC, it mainly \
            contains three optimizers, algorithm-specific arguments such as gamma and twin_critic, main and target \
            model. Especially, the ``auto_alpha`` mechanism for balancing max entropy target is also initialized here.
            This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._twin_critic = self._cfg.model.twin_critic

        # Weight Init for the last output layer
        init_w = self._cfg.learn.init_w
        self._model.actor_head[-1].mu.weight.data.uniform_(-init_w, init_w)
        self._model.actor_head[-1].mu.bias.data.uniform_(-init_w, init_w)
        self._model.actor_head[-1].log_sigma_layer.weight.data.uniform_(-init_w, init_w)
        self._model.actor_head[-1].log_sigma_layer.bias.data.uniform_(-init_w, init_w)

        self._optimizer_q = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_q,
        )
        self._optimizer_policy = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_policy,
        )

        # Algorithm-Specific Config
        self._gamma = self._cfg.learn.discount_factor
        if self._cfg.learn.auto_alpha:
            if self._cfg.learn.target_entropy is None:
                assert 'action_shape' in self._cfg.model, "SQILSACPolicy need network model with action_shape variable"
                self._target_entropy = -np.prod(self._cfg.model.action_shape)
            else:
                self._target_entropy = self._cfg.learn.target_entropy
            if self._cfg.learn.log_space:
                self._log_alpha = torch.log(torch.FloatTensor([self._cfg.learn.alpha]))
                self._log_alpha = self._log_alpha.to(self._device).requires_grad_()
                self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=self._cfg.learn.learning_rate_alpha)
                assert self._log_alpha.shape == torch.Size([1]) and self._log_alpha.requires_grad
                self._alpha = self._log_alpha.detach().exp()
                self._auto_alpha = True
                self._log_space = True
            else:
                self._alpha = torch.FloatTensor([self._cfg.learn.alpha]).to(self._device).requires_grad_()
                self._alpha_optim = torch.optim.Adam([self._alpha], lr=self._cfg.learn.learning_rate_alpha)
                self._auto_alpha = True
                self._log_space = False
        else:
            self._alpha = torch.tensor(
                [self._cfg.learn.alpha], requires_grad=False, device=self._device, dtype=torch.float32
            )
            self._auto_alpha = False

        # Main and target models
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

        # monitor cossimilarity and entropy switch
        self._monitor_cos = True
        self._monitor_entropy = True

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, action, priority.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For SAC, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys such as ``weight``.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The information dict that indicated training result, which will be \
                recorded in text log and tensorboard, values must be python scalar or a list of scalars. For the \
                detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            For SQIL + SAC, input data is composed of two parts with the same size: agent data and expert data. \
            Both of them are relabelled with new reward according to SQIL algorithm.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for SACPolicy: ``ding.policy.tests.test_sac``.
        """
        loss_dict = {}
        if self._monitor_cos:
            agent_data = default_preprocess_learn(
                data[0:len(data) // 2],
                use_priority=self._priority,
                use_priority_IS_weight=self._cfg.priority_IS_weight,
                ignore_done=self._cfg.learn.ignore_done,
                use_nstep=False
            )

            expert_data = default_preprocess_learn(
                data[len(data) // 2:],
                use_priority=self._priority,
                use_priority_IS_weight=self._cfg.priority_IS_weight,
                ignore_done=self._cfg.learn.ignore_done,
                use_nstep=False
            )
            if self._cuda:
                agent_data = to_device(agent_data, self._device)
                expert_data = to_device(expert_data, self._device)

        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        obs = data['obs']
        next_obs = data['next_obs']
        reward = data['reward']
        done = data['done']

        # 1. predict q value
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']

        # 2. predict target value
        with torch.no_grad():
            (mu, sigma) = self._learn_model.forward(next_obs, mode='compute_actor')['logit']
            dist = Independent(Normal(mu, sigma), 1)
            pred = dist.rsample()
            next_action = torch.tanh(pred)
            y = 1 - next_action.pow(2) + 1e-6
            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            next_log_prob = dist.log_prob(pred).unsqueeze(-1)
            next_log_prob = next_log_prob - torch.log(y).sum(-1, keepdim=True)

            next_data = {'obs': next_obs, 'action': next_action}
            target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
            # the value of a policy according to the maximum entropy objective
            if self._twin_critic:
                # find min one as target q value
                target_q_value = torch.min(target_q_value[0],
                                           target_q_value[1]) - self._alpha * next_log_prob.squeeze(-1)
            else:
                target_q_value = target_q_value - self._alpha * next_log_prob.squeeze(-1)

        # 3. compute q loss
        if self._twin_critic:
            q_data0 = v_1step_td_data(q_value[0], target_q_value, reward, done, data['weight'])
            loss_dict['critic_loss'], td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
            q_data1 = v_1step_td_data(q_value[1], target_q_value, reward, done, data['weight'])
            loss_dict['twin_critic_loss'], td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
            td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
        else:
            q_data = v_1step_td_data(q_value, target_q_value, reward, done, data['weight'])
            loss_dict['critic_loss'], td_error_per_sample = v_1step_td_error(q_data, self._gamma)

        # 4. update q network
        self._optimizer_q.zero_grad()
        if self._twin_critic:
            (loss_dict['critic_loss'] + loss_dict['twin_critic_loss']).backward()
        else:
            loss_dict['critic_loss'].backward()
        self._optimizer_q.step()

        # 5. evaluate to get action distribution
        if self._monitor_cos:
            # agent
            (mu, sigma) = self._learn_model.forward(agent_data['obs'], mode='compute_actor')['logit']
            dist = Independent(Normal(mu, sigma), 1)
            pred = dist.rsample()
            action = torch.tanh(pred)
            y = 1 - action.pow(2) + 1e-6
            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            agent_log_prob = dist.log_prob(pred).unsqueeze(-1)
            agent_log_prob = agent_log_prob - torch.log(y).sum(-1, keepdim=True)

            eval_data = {'obs': agent_data['obs'], 'action': action}
            agent_new_q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
            if self._twin_critic:
                agent_new_q_value = torch.min(agent_new_q_value[0], agent_new_q_value[1])
            # expert
            (mu, sigma) = self._learn_model.forward(expert_data['obs'], mode='compute_actor')['logit']
            dist = Independent(Normal(mu, sigma), 1)
            pred = dist.rsample()
            action = torch.tanh(pred)
            y = 1 - action.pow(2) + 1e-6
            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            expert_log_prob = dist.log_prob(pred).unsqueeze(-1)
            expert_log_prob = expert_log_prob - torch.log(y).sum(-1, keepdim=True)

            eval_data = {'obs': expert_data['obs'], 'action': action}
            expert_new_q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
            if self._twin_critic:
                expert_new_q_value = torch.min(expert_new_q_value[0], expert_new_q_value[1])

        (mu, sigma) = self._learn_model.forward(data['obs'], mode='compute_actor')['logit']
        dist = Independent(Normal(mu, sigma), 1)
        # for monitor the entropy of policy
        if self._monitor_entropy:
            dist_entropy = dist.entropy()
            entropy = dist_entropy.mean()

        pred = dist.rsample()
        action = torch.tanh(pred)
        y = 1 - action.pow(2) + 1e-6
        # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
        log_prob = dist.log_prob(pred).unsqueeze(-1)
        log_prob = log_prob - torch.log(y).sum(-1, keepdim=True)

        eval_data = {'obs': obs, 'action': action}
        new_q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
        if self._twin_critic:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])

        # 6. compute policy loss
        policy_loss = (self._alpha * log_prob - new_q_value.unsqueeze(-1)).mean()
        loss_dict['policy_loss'] = policy_loss

        # 7. update policy network
        if self._monitor_cos:
            agent_policy_loss = (self._alpha * agent_log_prob - agent_new_q_value.unsqueeze(-1)).mean()
            expert_policy_loss = (self._alpha * expert_log_prob - expert_new_q_value.unsqueeze(-1)).mean()
            loss_dict['agent_policy_loss'] = agent_policy_loss
            loss_dict['expert_policy_loss'] = expert_policy_loss
            self._optimizer_policy.zero_grad()
            loss_dict['agent_policy_loss'].backward()
            agent_grad = (list(list(self._learn_model.actor.children())[-1].children())[-1].weight.grad).mean()
            self._optimizer_policy.zero_grad()
            loss_dict['expert_policy_loss'].backward()
            expert_grad = (list(list(self._learn_model.actor.children())[-1].children())[-1].weight.grad).mean()
            cos = nn.CosineSimilarity(dim=0)
            cos_similarity = cos(agent_grad, expert_grad)
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()

        # 8. compute alpha loss
        if self._auto_alpha:
            if self._log_space:
                log_prob = log_prob + self._target_entropy
                loss_dict['alpha_loss'] = -(self._log_alpha * log_prob.detach()).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha = self._log_alpha.detach().exp()
            else:
                log_prob = log_prob + self._target_entropy
                loss_dict['alpha_loss'] = -(self._alpha * log_prob.detach()).mean()

                self._alpha_optim.zero_grad()
                loss_dict['alpha_loss'].backward()
                self._alpha_optim.step()
                self._alpha = max(0, self._alpha)

        loss_dict['total_loss'] = sum(loss_dict.values())

        # target update
        self._target_model.update(self._learn_model.state_dict())
        var_monitor = {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.detach().mean().item(),
            'agent_td_error': td_error_per_sample.detach().chunk(2, dim=0)[0].mean().item(),
            'expert_td_error': td_error_per_sample.detach().chunk(2, dim=0)[1].mean().item(),
            'alpha': self._alpha.item(),
            'target_q_value': target_q_value.detach().mean().item(),
            'mu': mu.detach().mean().item(),
            'sigma': sigma.detach().mean().item(),
            'q_value0': new_q_value[0].detach().mean().item(),
            'q_value1': new_q_value[1].detach().mean().item(),
            **loss_dict,
        }
        if self._monitor_cos:
            var_monitor['cos_similarity'] = cos_similarity.item()
        if self._monitor_entropy:
            var_monitor['entropy'] = entropy.item()
        return var_monitor

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        twin_critic = ['twin_critic_loss'] if self._twin_critic else []
        alpha_loss = ['alpha_loss'] if self._auto_alpha else []
        cos_similarity = ['cos_similarity'] if self._monitor_cos else []
        entropy = ['entropy'] if self._monitor_entropy else []
        return [
            'value_loss'
            'alpha_loss',
            'policy_loss',
            'critic_loss',
            'cur_lr_q',
            'cur_lr_p',
            'target_q_value',
            'alpha',
            'td_error',
            'agent_td_error',
            'expert_td_error',
            'mu',
            'sigma',
            'q_value0',
            'q_value1',
        ] + twin_critic + alpha_loss + cos_similarity + entropy
