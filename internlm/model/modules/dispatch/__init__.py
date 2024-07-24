# adapted from https://github.com/InternLM/xtuner/blob/main/xtuner/model/modules/dispatch/__init__.py

# Copyright (c) OpenMMLab. All rights reserved.
import os
import types
import importlib
import torch
import transformers

from collections import abc
from typing import Any, Optional, Type, Union
from packaging.version import Version as digit_version
from transformers.utils.import_utils import is_flash_attn_2_available


# adapted LazyObject from https://github.com/open-mmlab/mmengine/blob/main/mmengine/config/lazy.py
class LazyObject:
    def __init__(self,
                 module: Union[str, list, tuple],
                 imported: Optional[str] = None,
                 location: Optional[str] = None):
        if not isinstance(module, str) and not is_seq_of(module, str):
            raise TypeError('module should be `str`, `list`, or `tuple`'
                            f'but got {type(module)}, this might be '
                            'a bug of MMEngine, please report it to '
                            'https://github.com/open-mmlab/mmengine/issues')
        self._module: Union[str, list, tuple] = module

        if not isinstance(imported, str) and imported is not None:
            raise TypeError('imported should be `str` or None, but got '
                            f'{type(imported)}, this might be '
                            'a bug of MMEngine, please report it to '
                            'https://github.com/open-mmlab/mmengine/issues')
        self._imported = imported
        self.location = location

    def build(self) -> Any:
        if isinstance(self._module, str):
            try:
                module = importlib.import_module(self._module)
            except Exception as e:
                raise type(e)(f'Failed to import {self._module} '
                              f'in {self.location} for {e}')

            if self._imported is not None:
                if hasattr(module, self._imported):
                    module = getattr(module, self._imported)
                else:
                    raise ImportError(
                        f'Failed to import {self._imported} '
                        f'from {self._module} in {self.location}')

            return module
        else:
            try:
                for module in self._module:
                    importlib.import_module(module)  # type: ignore
                module_name = self._module[0].split('.')[0]
                return importlib.import_module(module_name)
            except Exception as e:
                raise type(e)(f'Failed to import {self.module} '
                              f'in {self.location} for {e}')

    @property
    def module(self):
        if isinstance(self._module, str):
            return self._module
        return self._module[0].split('.')[0]

    def __call__(self, *args, **kwargs):
        raise RuntimeError()

    def __deepcopy__(self, memo):
        return LazyObject(self._module, self._imported, self.location)

    def __getattr__(self, name):
        if self.location is not None:
            location = self.location.split(', line')[0]
        else:
            location = self.location
        return LazyAttr(name, self, location)

    def __str__(self) -> str:
        if self._imported is not None:
            return self._imported
        return self.module

    __repr__ = __str__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


# adapted LazyAttr from https://github.com/open-mmlab/mmengine/blob/main/mmengine/config/lazy.py
class LazyAttr:
    def __init__(self,
                 name: str,
                 source: Union['LazyObject', 'LazyAttr'],
                 location=None):
        self.name = name
        self.source: Union[LazyAttr, LazyObject] = source

        if isinstance(self.source, LazyObject):
            if isinstance(self.source._module, str):
                if self.source._imported is None:
                    self._module = self.source._module
                else:
                    self._module = f'{self.source._module}.{self.source}'
            else:
                self._module = str(self.source)
        elif isinstance(self.source, LazyAttr):
            self._module = f'{self.source._module}.{self.source.name}'
        self.location = location

    @property
    def module(self):
        return self._module

    def __call__(self, *args, **kwargs: Any) -> Any:
        raise RuntimeError()

    def __getattr__(self, name: str) -> 'LazyAttr':
        return LazyAttr(name, self)

    def __deepcopy__(self, memo):
        return LazyAttr(self.name, self.source)

    def build(self) -> Any:
        obj = self.source.build()
        try:
            return getattr(obj, self.name)
        except AttributeError:
            raise ImportError(f'Failed to import {self.module}.{self.name} in '
                              f'{self.location}')
        except ImportError as e:
            raise e

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


# adapt is_seq_of from https://github.com/open-mmlab/mmengine/blob/main/mmengine/utils/misc.py
def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Type = None) -> bool:
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


TRANSFORMERS_VERSION = digit_version(transformers.__version__)
IS_LOW_VERSION_TRANSFORMERS = TRANSFORMERS_VERSION < digit_version('4.38')
# Transformers requires torch version >= 2.1.1 when using Torch SDPA.
# Refer to https://github.com/huggingface/transformers/blob/caa5c65db1f4db617cdac2ad667ba62edf94dd98/src/transformers/modeling_utils.py#L1611  # noqa: E501
SUPPORT_FLASH1 = digit_version(torch.__version__) >= digit_version('2.1.1')
SUPPORT_FLASH2 = is_flash_attn_2_available()
SUPPORT_FLASH = SUPPORT_FLASH1 or SUPPORT_FLASH2

USE_TRITON_KERNEL = bool(os.getenv('USE_TRITON_KERNEL', default=0))
SUPPORT_TRITON = False
try:
    import triton  # pre-check # noqa: F401
    import triton.language as tl  # pre-check # noqa: F401
    SUPPORT_TRITON = True
except ImportError:
    if USE_TRITON_KERNEL:
        raise RuntimeError(
            'USE_TRITON_KERNEL is set to 1, but triton has not been installed.'
            ' Run `pip install triton==2.1.0` to install triton.')

NO_ATTN_WEIGHTS_MSG = (
    'Due to the implementation of the PyTorch version of flash attention, '
    'even when the `output_attentions` flag is set to True, it is not '
    'possible to return the `attn_weights`.')

LOWEST_TRANSFORMERS_VERSION = dict(
    InternLM2ForCausalLM=digit_version('4.36'),
    InternLMForCausalLM=digit_version('4.36'),
    LlamaForCausalLM=digit_version('4.36'),
    Phi3ForCausalLM=digit_version('4.39'),
    MistralForCausalLM=digit_version('4.36'),
    # Training mixtral with lower version may lead to nccl timeout
    # Refer to https://github.com/microsoft/DeepSpeed/issues/5066
    MixtralForCausalLM=digit_version('4.40'),
    DeepseekV2ForCausalLM=digit_version('4.40'),
)

ATTN_DISPATCH_MAPPING = dict(
    InternLMAttention=LazyObject('internlm.model.modules.dispatch.internlm',
                                 'internlm_attn_forward'),
    LlamaFlashAttention2=LazyObject('internlm.model.modules.dispatch.llama',
                                    'llama_attn_forward'),
)

ATTN_LEGACY_DISPATCH_MAPPING = dict(
    LlamaFlashAttention2=LazyObject('internlm.model.modules.dispatch.llama',
                                    'llama_attn_forward_legacy'), )

VARLEN_ATTN_DISPATCH_MAPPING = dict(
    InternLMAttention=LazyObject('internlm.model.modules.dispatch.internlm',
                                 'internlm_varlen_attn_forward'),
    LlamaFlashAttention2=LazyObject('internlm.model.modules.dispatch.llama',
                                    'llama_varlen_attn_forward'),
)

VARLEN_ATTN_LEGACY_DISPATCH_MAPPING = dict(
    LlamaFlashAttention2=LazyObject('internlm.model.modules.dispatch.llama',
                                    'llama_varlen_attn_forward_legacy'), )

RMS_DISPATCH_MAPPING = dict(
    InternLMRMSNorm=LazyObject('internlm.model.modules.dispatch.triton_kernels',
                               'rms_norm_forward'),
    LlamaRMSNorm=LazyObject('internlm.model.modules.dispatch.triton_kernels',
                            'rms_norm_forward'),
)

ROTE_DISPATCH_MAPPING = dict(
    InternLMRotaryEmbedding=LazyObject(
        'internlm.model.modules.dispatch.internlm', 'InternLMRotaryEmbedding'),
)


def log_once(func):
    logged = False

    def wrapper(*args, **kwargs):
        nonlocal logged
        if not logged:
            logged = True
            func(*args, **kwargs)
        return

    return wrapper


def dispatch_attn_forward(model):

    if not SUPPORT_FLASH2:
        return

    attn_forward = None
    for module in model.modules():
        name = type(module).__name__
        if (IS_LOW_VERSION_TRANSFORMERS
                and name in ATTN_LEGACY_DISPATCH_MAPPING):
            if attn_forward is None:
                attn_forward = ATTN_LEGACY_DISPATCH_MAPPING[name]
                attn_forward = attn_forward.build()
            module.forward = types.MethodType(attn_forward, module)
        elif name in ATTN_DISPATCH_MAPPING:
            if attn_forward is None:
                attn_forward = ATTN_DISPATCH_MAPPING[name]
                attn_forward = attn_forward.build()
            module.forward = types.MethodType(attn_forward, module)


def dispatch_varlen_attn_forward(model):

    if not SUPPORT_FLASH2:
        return

    varlen_attn_forward = None
    for module in model.modules():
        name = type(module).__name__
        if (IS_LOW_VERSION_TRANSFORMERS
                and name in VARLEN_ATTN_LEGACY_DISPATCH_MAPPING):
            if varlen_attn_forward is None:
                varlen_attn_forward = VARLEN_ATTN_LEGACY_DISPATCH_MAPPING[name]
                varlen_attn_forward = varlen_attn_forward.build()
            module.forward = types.MethodType(varlen_attn_forward, module)
        elif name in VARLEN_ATTN_DISPATCH_MAPPING:
            if varlen_attn_forward is None:
                varlen_attn_forward = VARLEN_ATTN_DISPATCH_MAPPING[name]
                varlen_attn_forward = varlen_attn_forward.build()
            module.forward = types.MethodType(varlen_attn_forward, module)


def dispatch_rmsnorm_forward(model):

    if (not SUPPORT_TRITON) or (not USE_TRITON_KERNEL):
        return

    rms_forward = None
    for module in model.modules():
        name = type(module).__name__
        if name in RMS_DISPATCH_MAPPING:
            if rms_forward is None:
                rms_forward = RMS_DISPATCH_MAPPING[name]
                rms_forward = rms_forward.build()
            module.forward = types.MethodType(rms_forward, module)


def replace_rote(model):

    def traverse(module):
        for name, child in module.named_children():
            cls_name = type(child).__name__
            if cls_name in ROTE_DISPATCH_MAPPING:
                assert hasattr(model.config, 'rope_theta'), \
                    '`rope_theta` should be in the model config.'
                rope_theta = model.config.rope_theta

                rote = ROTE_DISPATCH_MAPPING[cls_name]
                rote = rote.build()
                dim_model = child.inv_freq.shape[0] * 2
                child_new = rote(dim_model, child.max_seq_len_cached,
                                 rope_theta).to(
                                     device=child.inv_freq.device,
                                     dtype=child.inv_freq.dtype)
                setattr(module, name, child_new)
            else:
                traverse(child)

    traverse(model)


def dispatch_modules(model, use_varlen_attn=False):

    def check(model_name):
        assert 'ForCausalLM' in model_name
        msg = '{} requires transformers version at least {}, but got {}'
        if model_name in LOWEST_TRANSFORMERS_VERSION:
            assert TRANSFORMERS_VERSION >= LOWEST_TRANSFORMERS_VERSION[
                model_name], msg.format(
                    model_name, LOWEST_TRANSFORMERS_VERSION[model_name],
                    TRANSFORMERS_VERSION)

    check(type(model).__name__)
    if use_varlen_attn:
        dispatch_varlen_attn_forward(model)
    else:
        dispatch_attn_forward(model)
    dispatch_rmsnorm_forward(model)
    replace_rote(model)


__all__ = ['dispatch_modules']
