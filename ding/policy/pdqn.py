from typing import List, Dict, Any, Tuple
from collections import namedtuple
import copy
import torch

from ding.torch_utils import Adam, to_device
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error, get_nstep_return_data, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('pdqn')
class PDQNPolicy(Policy):
    """
    Overview:
        Policy class of PDQN algorithm, which extends the DQN algorithm on discrete-continuous hybrid action spaces.
        Paper link: https://arxiv.org/abs/1810.06394.

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      pdqn           | RL policy register name, refer to      | This arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | This arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy  | This value is always
                                                        | or off-policy                          | False for PDQN
        4  ``priority``         bool     False          | Whether use priority(PER)              | Priority sample,
                                                                                                 | update priority
        5  | ``priority_IS``    bool     False          | Whether use Importance Sampling Weight
           | ``_weight``                                | to correct biased update. If True,
                                                        | priority must be True.
        6  | ``discount_``      float    0.97,          | Reward's future discount factor, aka.  | May be 1 when sparse
           | ``factor``                  [0.95, 0.999]  | gamma                                  | reward env

        7  ``nstep``            int      1,             | N-step reward discount sum for target
                                         [3, 5]         | q_value estimation
        8  | ``learn.update``   int      3              | How many updates(iterations) to train  | This args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        9  | ``learn.batch_``   int      64             | The number of samples of an iteration
           | ``size``
           | ``_gpu``
        11 | ``learn.learning`` float    0.001          | Gradient step length of an iteration.
           | ``_rate``
        12 | ``learn.target_``  int      100            | Frequence of target network update.    | Hard(assign) update
           | ``update_freq``
        13 | ``learn.ignore_``  bool     False          | Whether ignore done for target value   | Enable it for some
           | ``done``                                   | calculation.                           | fake termination env
        14 ``collect.n_sample`` int      [8, 128]       | The number of training samples of a    | It varies from
                                                        | call of collector.                     | different envs
        15 | ``collect.unroll`` int      1              | unroll length of an iteration          | In RNN, unroll_len>1
           | ``_len``
        16 | ``collect.noise``  float    0.1            | add noise to continuous args
           | ``_sigma``                                 | during collection
        17 | ``other.eps.type`` str      exp            | exploration rate decay type            | Support ['exp',
                                                                                                 | 'linear'].
        18 | ``other.eps.``     float    0.95           | start value of exploration rate        | [0,1]
           | ``start``
        19 | ``other.eps.``     float    0.05           | end value of exploration rate          | [0,1]
           | ``end``
        20 | ``other.eps.``     int      10000          | decay length of exploration            | greater than 0. set
           | ``decay``                                                                           | decay=10000 means
                                                                                                 | the exploration rate
                                                                                                 | decay from start
                                                                                                 | value to end value
                                                                                                 | during decay length.
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='pdqn',
        # (bool) Whether to use cuda in policy.
        cuda=False,
        # (bool) Whether learning policy is the same as collecting data policy(on-policy).
        on_policy=False,
        # (bool) Whether to enable priority experience sample.
        priority=False,
        # (bool) Whether to use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (float) Discount factor(gamma) for returns.
        discount_factor=0.97,
        # (int) The number of step for calculating target q_value.
        nstep=1,
        # learn_mode config
        learn=dict(
            # (int) How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,
            # (int) How many samples in a training batch.
            batch_size=64,
            # (float) The step size of gradient descent.
            learning_rate=0.001,
            # (int) Frequence of target network update.
            target_theta=0.005,
            # (bool) Whether ignore done(usually for max step termination env).
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] shoule be set.
            # n_sample=8,
            # (int) Split episodes or trajectories into pieces with length `unroll_len`.
            unroll_len=1,
            # (float) It is a must to add noise during collection. So here omits noise and only set ``noise_sigma``.
            noise_sigma=0.1,
        ),
        eval=dict(),  # for compatibility
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                # (float) Epsilon start value.
                start=0.95,
                # (float) Epsilon end value.
                end=0.1,
                # (int) Decay length(env step)
                decay=10000,
            ),
            replay_buffer=dict(
                # (int) Maximum size of replay buffer. Usually, larger buffer size is better.
                replay_buffer_size=10000,
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

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For example about PDQN, its registered name is ``pdqn`` and the import_names is \
            ``ding.model.template.pdqn``.
        """
        return 'pdqn', ['ding.model.template.pdqn']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For PDQN, it mainly \
            contains two optimizers, algorithm-specific arguments such as nstep and gamma, main and target model.
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
        # Optimizer
        self._dis_optimizer = Adam(
            list(self._model.dis_head.parameters()) + list(self._model.cont_encoder.parameters()),
            # this is very important to put cont_encoder.parameters in here.
            lr=self._cfg.learn.learning_rate_dis
        )
        self._cont_optimizer = Adam(list(self._model.cont_head.parameters()), lr=self._cfg.learn.learning_rate_cont)

        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='hybrid_argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()
        self.cont_train_cnt = 0
        self.disc_train_cnt = 0
        self.train_cnt = 0

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, q value, target_q_value, priority.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For PDQN, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys such as ``weight`` \
                and ``value_gamma``.
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
            For more detailed examples, please refer to our unittest for PDQNPolicy: ``ding.policy.tests.test_pdqn``.
        """
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)

        self.train_cnt += 1
        # ================================
        # Continuous args network forward
        # ================================
        if self.train_cnt == 1 or self.train_cnt % self._cfg.learn.update_circle in range(5, 10):
            dis_loss = torch.Tensor([0])
            td_error_per_sample = torch.Tensor([0])
            target_q_value = torch.Tensor([0])

            action_args = self._learn_model.forward(data['obs'], mode='compute_continuous')['action_args']

            # Current q value (main model) for cont loss
            discrete_inputs = {'state': data['obs'], 'action_args': action_args}
            # with torch.no_grad():
            q_pi_action_value = self._learn_model.forward(discrete_inputs, mode='compute_discrete')['logit']
            cont_loss = -q_pi_action_value.sum(dim=-1).mean()

            # ================================
            # Continuous args network update
            # ================================
            self._cont_optimizer.zero_grad()
            cont_loss.backward()
            self._cont_optimizer.step()

        # ====================
        # Q-learning forward
        # ====================
        if self.train_cnt == 1 or self.train_cnt % self._cfg.learn.update_circle in range(0, 5):
            cont_loss = torch.Tensor([0])
            q_pi_action_value = torch.Tensor([0])
            self._learn_model.train()
            self._target_model.train()
            # Current q value (main model)
            discrete_inputs = {'state': data['obs'], 'action_args': data['action']['action_args']}
            q_data_action_args_value = self._learn_model.forward(discrete_inputs, mode='compute_discrete')['logit']

            # Target q value
            with torch.no_grad():
                next_action_args = self._learn_model.forward(data['next_obs'], mode='compute_continuous')['action_args']
                next_action_args_cp = next_action_args.clone().detach()
                next_discrete_inputs = {'state': data['next_obs'], 'action_args': next_action_args_cp}
                target_q_value = self._target_model.forward(next_discrete_inputs, mode='compute_discrete')['logit']
                # Max q value action (main model)
                target_q_discrete_action = self._learn_model.forward(
                    next_discrete_inputs, mode='compute_discrete'
                )['action']['action_type']

            data_n = q_nstep_td_data(
                q_data_action_args_value, target_q_value, data['action']['action_type'], target_q_discrete_action,
                data['reward'], data['done'], data['weight']
            )
            value_gamma = data.get('value_gamma')
            dis_loss, td_error_per_sample = q_nstep_td_error(
                data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma
            )

            # ====================
            # Q-learning update
            # ====================
            self._dis_optimizer.zero_grad()
            dis_loss.backward()
            self._dis_optimizer.step()

            # =============
            # after update
            # =============
            self._target_model.update(self._learn_model.state_dict())

        return {
            'cur_lr': self._dis_optimizer.defaults['lr'],
            'q_loss': dis_loss.item(),
            'total_loss': cont_loss.item() + dis_loss.item(),
            'continuous_loss': cont_loss.item(),
            'q_value': q_pi_action_value.mean().item(),
            'priority': td_error_per_sample.abs().tolist(),
            'reward': data['reward'].mean().item(),
            'target_q_value': target_q_value.mean().item(),
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model, target model, discrete part optimizer, and \
            continuous part optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'dis_optimizer': self._dis_optimizer.state_dict(),
            'cont_optimizer': self._cont_optimizer.state_dict()
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._dis_optimizer.load_state_dict(state_dict['dis_optimizer'])
        self._cont_optimizer.load_state_dict(state_dict['cont_optimizer'])

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For PDQN, it contains the \
            collect_model to balance the exploration and exploitation with epsilon-greedy sample mechanism and \
            continuous action mechanism, besides, other algorithm-specific arguments such as unroll_len and nstep are \
            also initialized here.
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.

        .. tip::
            Some variables need to initialize independently in different modes, such as gamma and nstep in PDQN. This \
            design is for the convenience of parallel execution of different policy modes.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.discount_factor  # necessary for parallel
        self._nstep = self._cfg.nstep  # necessary for parallel
        self._collect_model = model_wrap(
            self._model,
            wrapper_name='action_noise',
            noise_type='gauss',
            noise_kwargs={
                'mu': 0.0,
                'sigma': self._cfg.collect.noise_sigma
            },
            noise_range=None
        )
        self._collect_model = model_wrap(self._collect_model, wrapper_name='hybrid_eps_greedy_multinomial_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
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
            For more detailed examples, please refer to our unittest for PDQNPolicy: ``ding.policy.tests.test_pdqn``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            action_args = self._collect_model.forward(data, 'compute_continuous', eps=eps)['action_args']
            inputs = {'state': data, 'action_args': action_args.clone().detach()}
            output = self._collect_model.forward(inputs, 'compute_discrete', eps=eps)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _get_train_sample(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. In PDQN, a train sample is a processed transition. \
            This method is usually used in collectors to execute necessary \
            RL data preprocessing before training, which can help learner amortize revelant time consumption. \
            In addition, you can also implement this method as an identity function and do the data processing \
            in ``self._forward_learn`` method.
        Arguments:
            - transitions (:obj:`List[Dict[str, Any]`): The trajectory data (a list of transition), each element is \
                the same format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each element is the similar format \
                as input transitions, but may contain more data for training, such as nstep reward and target obs.
        """
        transitions = get_nstep_return_data(transitions, self._nstep, gamma=self._gamma)
        return get_train_sample(transitions, self._unroll_len)

    def _process_transition(self, obs: torch.Tensor, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For PDQN, it contains obs, next_obs, action, reward, done and logit.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For PDQN, it contains the hybrid action and the logit (discrete part q_value) of the action.
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

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For PDQN, it contains the \
            eval model to greedily select action with argmax q_value mechanism.
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
            For more detailed examples, please refer to our unittest for PDQNPolicy: ``ding.policy.tests.test_pdqn``.
        """
        # data_id = list(data.keys())
        # data = default_collate(list(data.values()))
        # if self._cuda:
            # data = to_device(data, self._device)
        # self._eval_model.eval()
        # with torch.no_grad():
            # action_args = self._eval_model.forward(data, mode='compute_continuous')['action_args']
            # inputs = {'state': data, 'action_args': action_args.clone().detach()}
            # output = self._eval_model.forward(inputs, mode='compute_discrete')
        # if self._cuda:
            # output = to_device(output, 'cpu')
        # output = default_decollate(output)
        # return {i: d for i, d in zip(data_id, output)}
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data)
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
        return ['cur_lr', 'total_loss', 'q_loss', 'continuous_loss', 'q_value', 'reward', 'target_q_value']
