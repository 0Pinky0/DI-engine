from typing import Union, Mapping, List, NamedTuple, Tuple, Callable, Optional, Any, Dict
import copy
from ditk import logging
import random
from functools import lru_cache  # in python3.9, we can change to cache
import numpy as np
import torch
import treetensor.torch as ttorch


def get_shape0(data: Union[List, Dict, torch.Tensor, ttorch.Tensor]) -> int:
    """
    Overview:
        Get shape[0] of data's torch tensor or treetensor
    Arguments:
        - data (:obj:`Union[List,Dict,torch.Tensor,ttorch.Tensor]`): data to be analysed
    Returns:
        - shape[0] (:obj:`int`): first dimension length of data, usually the batchsize.
    """
    if isinstance(data, list) or isinstance(data, tuple):
        return get_shape0(data[0])
    elif isinstance(data, dict):
        for k, v in data.items():
            return get_shape0(v)
    elif isinstance(data, torch.Tensor):
        return data.shape[0]
    elif isinstance(data, ttorch.Tensor):

        def fn(t):
            item = list(t.values())[0]
            if np.isscalar(item[0]):
                return item[0]
            else:
                return fn(item)

        return fn(data.shape)
    else:
        raise TypeError("Error in getting shape0, not support type: {}".format(data))


def lists_to_dicts(
        data: Union[List[Union[dict, NamedTuple]], Tuple[Union[dict, NamedTuple]]],
        recursive: bool = False,
) -> Union[Mapping[object, object], NamedTuple]:
    """
    Overview:
        Transform a list of dicts to a dict of lists.
    Arguments:
        - data (:obj:`Union[List[Union[dict, NamedTuple]], Tuple[Union[dict, NamedTuple]]]`):
            A dict of lists need to be transformed
        - recursive (:obj:`bool`): whether recursively deals with dict element
    Returns:
        - newdata (:obj:`Union[Mapping[object, object], NamedTuple]`): A list of dicts as a result
    Example:
        >>> from ding.utils import *
        >>> lists_to_dicts([{1: 1, 10: 3}, {1: 2, 10: 4}])
        {1: [1, 2], 10: [3, 4]}
    """
    if len(data) == 0:
        raise ValueError("empty data")
    if isinstance(data[0], dict):
        if recursive:
            new_data = {}
            for k in data[0].keys():
                if isinstance(data[0][k], dict) and k != 'prev_state':
                    tmp = [data[b][k] for b in range(len(data))]
                    new_data[k] = lists_to_dicts(tmp)
                else:
                    new_data[k] = [data[b][k] for b in range(len(data))]
        else:
            new_data = {k: [data[b][k] for b in range(len(data))] for k in data[0].keys()}
    elif isinstance(data[0], tuple) and hasattr(data[0], '_fields'):  # namedtuple
        new_data = type(data[0])(*list(zip(*data)))
    else:
        raise TypeError("not support element type: {}".format(type(data[0])))
    return new_data


def dicts_to_lists(data: Mapping[object, List[object]]) -> List[Mapping[object, object]]:
    """
    Overview:
        Transform a dict of lists to a list of dicts.

    Arguments:
        - data (:obj:`Mapping[object, list]`): A list of dicts need to be transformed

    Returns:
        - newdata (:obj:`List[Mapping[object, object]]`): A dict of lists as a result

    Example:
        >>> from ding.utils import *
        >>> dicts_to_lists({1: [1, 2], 10: [3, 4]})
        [{1: 1, 10: 3}, {1: 2, 10: 4}]
    """
    new_data = [v for v in data.values()]
    new_data = [{k: v for k, v in zip(data.keys(), t)} for t in list(zip(*new_data))]
    return new_data


def override(cls: type) -> Callable[[
        Callable,
], Callable]:
    """
    Overview:
        Annotation for documenting method overrides.

    Arguments:
        - cls (:obj:`type`): The superclass that provides the overridden method. If this
            cls does not actually have the method, an error is raised.
    """

    def check_override(method: Callable) -> Callable:
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(method, cls))
        return method

    return check_override


def squeeze(data: object) -> object:
    """
    Overview:
        Squeeze data from tuple, list or dict to single object
    Arguments:
        - data (:obj:`object`): data to be squeezed
    Example:
        >>> a = (4, )
        >>> a = squeeze(a)
        >>> print(a)
        >>> 4
    """
    if isinstance(data, tuple) or isinstance(data, list):
        if len(data) == 1:
            return data[0]
        else:
            return tuple(data)
    elif isinstance(data, dict):
        if len(data) == 1:
            return list(data.values())[0]
    return data


default_get_set = set()


def default_get(
        data: dict,
        name: str,
        default_value: Optional[Any] = None,
        default_fn: Optional[Callable] = None,
        judge_fn: Optional[Callable] = None
) -> Any:
    """
    Overview:
        Getting the value by input, checks generically on the inputs with \
        at least ``data`` and ``name``. If ``name`` exists in ``data``, \
        get the value at ``name``; else, add ``name`` to ``default_get_set``\
        with value generated by \
        ``default_fn`` (or directly as ``default_value``) that \
        is checked by `` judge_fn`` to be legal.
    Arguments:
        - data(:obj:`dict`): Data input dictionary
        - name(:obj:`str`): Key name
        - default_value(:obj:`Optional[Any]`) = None,
        - default_fn(:obj:`Optional[Callable]`) = Value
        - judge_fn(:obj:`Optional[Callable]`) = None
    Returns:
        - ret(:obj:`list`): Splitted data
        - residual(:obj:`list`): Residule list
    """
    if name in data:
        return data[name]
    else:
        assert default_value is not None or default_fn is not None
        value = default_fn() if default_fn is not None else default_value
        if judge_fn:
            assert judge_fn(value), "defalut value({}) is not accepted by judge_fn".format(type(value))
        if name not in default_get_set:
            logging.warning("{} use default value {}".format(name, value))
            default_get_set.add(name)
        return value


def list_split(data: list, step: int) -> List[list]:
    """
    Overview:
        Split list of data by step.
    Arguments:
        - data(:obj:`list`): List of data for spliting
        - step(:obj:`int`): Number of step for spliting
    Returns:
        - ret(:obj:`list`): List of splitted data.
        - residual(:obj:`list`): Residule list. This value is ``None`` when  ``data`` divides ``steps``.
    Example:
        >>> list_split([1,2,3,4],2)
        ([[1, 2], [3, 4]], None)
        >>> list_split([1,2,3,4],3)
        ([[1, 2, 3]], [4])
    """
    if len(data) < step:
        return [], data
    ret = []
    divide_num = len(data) // step
    for i in range(divide_num):
        start, end = i * step, (i + 1) * step
        ret.append(data[start:end])
    if divide_num * step < len(data):
        residual = data[divide_num * step:]
    else:
        residual = None
    return ret, residual


def error_wrapper(fn, default_ret, warning_msg=""):
    """
    Overview:
        wrap the function, so that any Exception in the function will be catched and return the default_ret
    Arguments:
        - fn (:obj:`Callable`): the function to be wraped
        - default_ret (:obj:`obj`): the default return when an Exception occurred in the function
    Returns:
        - wrapper (:obj:`Callable`): the wrapped function
    Examples:
        >>> # Used to checkfor Fakelink (Refer to utils.linklink_dist_helper.py)
        >>> def get_rank():  # Get the rank of linklink model, return 0 if use FakeLink.
        >>>    if is_fake_link:
        >>>        return 0
        >>>    return error_wrapper(link.get_rank, 0)()
    """

    def wrapper(*args, **kwargs):
        try:
            ret = fn(*args, **kwargs)
        except Exception as e:
            ret = default_ret
            if warning_msg != "":
                one_time_warning(warning_msg, "\ndefault_ret = {}\terror = {}".format(default_ret, e))
        return ret

    return wrapper


class LimitedSpaceContainer:
    """
    Overview:
        A space simulator.
    Interfaces:
        ``__init__``, ``get_residual_space``, ``release_space``
    """

    def __init__(self, min_val: int, max_val: int) -> None:
        """
        Overview:
            Set ``min_val`` and ``max_val`` of the container, also set ``cur`` to ``min_val`` for initialization.
        Arguments:
            - min_val (:obj:`int`): Min volume of the container, usually 0.
            - max_val (:obj:`int`): Max volume of the container.
        """
        self.min_val = min_val
        self.max_val = max_val
        assert (max_val >= min_val)
        self.cur = self.min_val

    def get_residual_space(self) -> int:
        """
        Overview:
            Get all residual pieces of space. Set ``cur`` to ``max_val``
        Arguments:
            - ret (:obj:`int`): Residual space, calculated by ``max_val`` - ``cur``.
        """
        ret = self.max_val - self.cur
        self.cur = self.max_val
        return ret

    def acquire_space(self) -> bool:
        """
        Overview:
            Try to get one pice of space. If there is one, return True; Otherwise return False.
        Returns:
            - flag (:obj:`bool`): Whether there is any piece of residual space.
        """
        if self.cur < self.max_val:
            self.cur += 1
            return True
        else:
            return False

    def release_space(self) -> None:
        """
        Overview:
            Release only one piece of space. Decrement ``cur``, but ensure it won't be negative.
        """
        self.cur = max(self.min_val, self.cur - 1)

    def increase_space(self) -> None:
        """
        Overview:
            Increase one piece in space. Increment ``max_val``.
        """
        self.max_val += 1

    def decrease_space(self) -> None:
        """
        Overview:
            Decrease one piece in space. Decrement ``max_val``.
        """
        self.max_val -= 1


def deep_merge_dicts(original: dict, new_dict: dict) -> dict:
    """
    Overview:
        Merge two dicts by calling ``deep_update``
    Arguments:
        - original (:obj:`dict`): Dict 1.
        - new_dict (:obj:`dict`): Dict 2.
    Returns:
        - merged_dict (:obj:`dict`): A new dict that is d1 and d2 deeply merged.
    """
    original = original or {}
    new_dict = new_dict or {}
    merged = copy.deepcopy(original)
    if new_dict:  # if new_dict is neither empty dict nor None
        deep_update(merged, new_dict, True, [])
    return merged


def deep_update(
    original: dict,
    new_dict: dict,
    new_keys_allowed: bool = False,
    whitelist: Optional[List[str]] = None,
    override_all_if_type_changes: Optional[List[str]] = None
):
    """
    Overview:
        Update original dict with values from new_dict recursively.
    Arguments:
        - original (:obj:`dict`): Dictionary with default values.
        - new_dict (:obj:`dict`): Dictionary with values to be updated
        - new_keys_allowed (:obj:`bool`): Whether new keys are allowed.
        - whitelist (:obj:`Optional[List[str]]`):
            List of keys that correspond to dict
            values where new subkeys can be introduced. This is only at the top
            level.
        - override_all_if_type_changes(:obj:`Optional[List[str]]`):
            List of top level
            keys with value=dict, for which we always simply override the
            entire value (:obj:`dict`), if the "type" key in that value dict changes.

    .. note::

        If new key is introduced in new_dict, then if new_keys_allowed is not
        True, an error will be thrown. Further, for sub-dicts, if the key is
        in the whitelist, then new subkeys can be introduced.
    """
    whitelist = whitelist or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise RuntimeError("Unknown config parameter `{}`. Base config have: {}.".format(k, original.keys()))

        # Both original value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                    "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Whitelisted key -> ok to add new subkeys.
            elif k in whitelist:
                deep_update(original[k], value, True)
            # Non-whitelisted key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original


def flatten_dict(data: dict, delimiter: str = "/") -> dict:
    """
    Overview:
        Flatten the dict, see example
    Arguments:
        - data (:obj:`dict`): Original nested dict
        - delimiter (str): Delimiter of the keys of the new dict
    Returns:
        - data (:obj:`dict`): Flattened nested dict
    Example:
        >>> a
        {'a': {'b': 100}}
        >>> flatten_dict(a)
        {'a/b': 100}
    """
    data = copy.deepcopy(data)
    while any(isinstance(v, dict) for v in data.values()):
        remove = []
        add = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    add[delimiter.join([key, subkey])] = v
                remove.append(key)
        data.update(add)
        for k in remove:
            del data[k]
    return data


def set_pkg_seed(seed: int, use_cuda: bool = True) -> None:
    """
    Overview:
        Side effect function to set seed for ``random``, ``numpy random``, and ``torch's manual seed``.\
        This is usaually used in entry scipt in the section of setting random seed for all package and instance
    Argument:
        - seed(:obj:`int`): Set seed
        - use_cuda(:obj:`bool`) Whether use cude
    Examples:
        >>> # ../entry/xxxenv_xxxpolicy_main.py
        >>> ...
        # Set random seed for all package and instance
        >>> collector_env.seed(seed)
        >>> evaluator_env.seed(seed, dynamic_seed=False)
        >>> set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
        >>> ...
        # Set up RL Policy, etc.
        >>> ...

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


@lru_cache()
def one_time_warning(warning_msg: str) -> None:
    """
    Overview:
        Print warning message only once.
    Arguments:
        - warning_msg (:obj:`str`): Warning message.
    """

    logging.warning(warning_msg)


def split_fn(data, indices, start, end):
    """
    Overview:
        Split data by indices
    Arguments:
        - data (:obj:`Union[List, Dict, torch.Tensor, ttorch.Tensor]`): data to be analysed
        - indices (:obj:`np.ndarray`): indices to split
        - start (:obj:`int`): start index
        - end (:obj:`int`): end index
    """

    if data is None:
        return None
    elif isinstance(data, list):
        return [split_fn(d, indices, start, end) for d in data]
    elif isinstance(data, dict):
        return {k1: split_fn(v1, indices, start, end) for k1, v1 in data.items()}
    elif isinstance(data, str):
        return data
    else:
        return data[indices[start:end]]


def split_data_generator(data: dict, split_size: int, shuffle: bool = False) -> dict:
    """
    Overview:
        Split data into batches
    Arguments:
        - data (:obj:`dict`): data to be analysed
        - split_size (:obj:`int`): split size
        - shuffle (:obj:`bool`): whether shuffle
    """

    assert isinstance(data, dict), type(data)
    length = []
    for k, v in data.items():
        if v is None:
            continue
        elif k in ['prev_state', 'prev_actor_state', 'prev_critic_state']:
            length.append(len(v))
        elif isinstance(v, list) or isinstance(v, tuple):
            if isinstance(v[0], str):
                # some buffer data contains useless string infos, such as 'buffer_id',
                # which should not be split, so we just skip it
                continue
            else:
                length.append(get_shape0(v[0]))
        elif isinstance(v, dict):
            length.append(len(v[list(v.keys())[0]]))
        else:
            length.append(len(v))
    assert len(length) > 0
    # assert len(set(length)) == 1, "data values must have the same length: {}".format(length)
    # if continuous action, data['logit'] is list of length 2
    length = length[0]
    assert split_size >= 1
    if shuffle:
        indices = np.random.permutation(length)
    else:
        indices = np.arange(length)
    for i in range(0, length, split_size):
        if i + split_size > length:
            i = length - split_size
        batch = split_fn(data, indices, i, i + split_size)
        yield batch


class RunningMeanStd(object):
    """
    Overview:
       Wrapper to update new variable, new mean, and new count
    Interfaces:
        ``__init__``, ``update``, ``reset``, ``new_shape``
    Properties:
        - ``mean``, ``std``, ``_epsilon``, ``_shape``, ``_mean``, ``_var``, ``_count``
    """

    def __init__(self, epsilon=1e-4, shape=(), device=torch.device('cpu')):
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate  \
                signature; setup the properties.
        Arguments:
            - env (:obj:`gym.Env`): the environment to wrap.
            - epsilon (:obj:`Float`): the epsilon used for self for the std output
            - shape (:obj: `np.array`): the np array shape used for the expression  \
                of this wrapper on attibutes of mean and variance
        """
        self._epsilon = epsilon
        self._shape = shape
        self._device = device
        self.reset()

    def update(self, x):
        """
        Overview:
            Update mean, variable, and count
        Arguments:
            - ``x``: the batch
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_count = batch_count + self._count
        mean_delta = batch_mean - self._mean
        new_mean = self._mean + mean_delta * batch_count / new_count
        # this method for calculating new variable might be numerically unstable
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(mean_delta) * self._count * batch_count / new_count
        new_var = m2 / new_count
        self._mean = new_mean
        self._var = new_var
        self._count = new_count

    def reset(self):
        """
        Overview:
            Resets the state of the environment and reset properties: ``_mean``, ``_var``, ``_count``
        """
        if len(self._shape) > 0:
            self._mean = np.zeros(self._shape, 'float32')
            self._var = np.ones(self._shape, 'float32')
        else:
            self._mean, self._var = 0., 1.
        self._count = self._epsilon

    @property
    def mean(self) -> np.ndarray:
        """
        Overview:
            Property ``mean`` gotten  from ``self._mean``
        """
        if np.isscalar(self._mean):
            return self._mean
        else:
            return torch.FloatTensor(self._mean).to(self._device)

    @property
    def std(self) -> np.ndarray:
        """
        Overview:
            Property ``std`` calculated  from ``self._var`` and the epsilon value of ``self._epsilon``
        """
        std = np.sqrt(self._var + 1e-8)
        if np.isscalar(std):
            return std
        else:
            return torch.FloatTensor(std).to(self._device)

    @staticmethod
    def new_shape(obs_shape, act_shape, rew_shape):
        """
        Overview:
           Get new shape of observation, acton, and reward; in this case unchanged.
        Arguments:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        Returns:
            obs_shape (:obj:`Any`), act_shape (:obj:`Any`), rew_shape (:obj:`Any`)
        """
        return obs_shape, act_shape, rew_shape


def make_key_as_identifier(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overview:
        Make the key of dict into legal python identifier string so that it is
        compatible with some python magic method such as ``__getattr``.
    Arguments:
        - data (:obj:`Dict[str, Any]`): The original dict data.
    Return:
        - new_data (:obj:`Dict[str, Any]`): The new dict data with legal identifier keys.
    """

    def legalization(s: str) -> str:
        if s[0].isdigit():
            s = '_' + s
        return s.replace('.', '_')

    new_data = {}
    for k in data:
        new_k = legalization(k)
        new_data[new_k] = data[k]
    return new_data


def remove_illegal_item(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overview:
        Remove illegal item in dict info, like str, which is not compatible with Tensor.
    Arguments:
        - data (:obj:`Dict[str, Any]`): The original dict data.
    Return:
        - new_data (:obj:`Dict[str, Any]`): The new dict data without legal items.
    """
    new_data = {}
    for k, v in data.items():
        if isinstance(v, str):
            continue
        new_data[k] = data[k]
    return new_data
