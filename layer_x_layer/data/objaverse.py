# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import torch
import random


import collections
import multiprocessing
import pathlib
from enum import Enum

import fvdb
import torch
from numpy.random import RandomState
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

from utils import exp

from utils.Dataspec import DatasetSpec


class RandomSafeDataset(Dataset):
    """
    A dataset class that provides a deterministic random seed.
    However, in order to have consistent validation set, we need to set is_val=True for validation/test sets.
    Usage: First, inherent this class.
           Then, at the beginning of your get_item call, get an rng;
           Last, use this rng as the random state for your program.
    """

    def __init__(self, seed: int, _is_val: bool = False, skip_on_error: bool = False):
        self._seed = seed
        self._is_val = _is_val
        self.skip_on_error = skip_on_error
        if not self._is_val:
            self._manager = multiprocessing.Manager()
            self._read_count = self._manager.dict()
            self._rc_lock = multiprocessing.Lock()

    def get_rng(self, idx):
        if self._is_val:
            return RandomState(self._seed)
        with self._rc_lock:
            if idx not in self._read_count:
                self._read_count[idx] = 0
            rng = RandomState(exp.deterministic_hash((idx, self._read_count[idx], self._seed)))
            self._read_count[idx] += 1
        return rng

    def sanitize_specs(self, old_spec, available_spec):
        old_spec = set(old_spec)
        available_spec = set(available_spec)
        for os in old_spec:
            assert isinstance(os, DatasetSpec)
        new_spec = old_spec.intersection(available_spec)
        # lack_spec = old_spec.difference(new_spec)
        # if len(lack_spec) > 0:
        #     exp.logger.warning(f"Lack spec {lack_spec}.")
        return new_spec

    def _get_item(self, data_id, rng):
        raise NotImplementedError

    def __getitem__(self, data_id):
        rng = self.get_rng(data_id)
        if self.skip_on_error:
            try:
                return self._get_item(data_id, rng)
            except ConnectionAbortedError:
                return self.__getitem__(rng.randint(0, len(self) - 1))
            except Exception:
                # Just return a random other item.
                print(f"Warning: Get item {data_id} error, but handled.")
                return self.__getitem__(rng.randint(0, len(self) - 1))
        else:
            try:
                return self._get_item(data_id, rng)
            except ConnectionAbortedError:
                return self.__getitem__(rng.randint(0, len(self) - 1))


def list_collate(batch):
    """
    This just do not stack batch dimension.
    """
    from fvdb import GridBatch, JaggedTensor

    elem = None
    for e in batch:
        if e is not None:
            elem = e
            break
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return list_collate([torch.as_tensor(b) if b is not None else None for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, DictConfig) or isinstance(elem, ListConfig):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: list_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [list_collate(samples) for samples in transposed]
    elif isinstance(elem, GridBatch):
        return fvdb.cat(batch)
    
    return batch

class ObjaverseDataset(RandomSafeDataset):
    def __init__(self, onet_base_path, spec, split, resolution, image_base_path=None,
                 random_seed=0, hparams=None, skip_on_error=False, custom_name="objaverse",
                 text_emb_path="../data/objaverse/objaverse/text_emb", null_embed_path="./assets/null_text_emb.pkl", text_embed_drop_prob=0.0, max_text_len=77,
                 duplicate_num=1, split_base_path=None, **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)

        self.skip_on_error = skip_on_error
        self.custom_name = custom_name
        self.resolution = resolution
        self.split = split
        self.spec = spec
        
        # setup path
        self.onet_base_path = onet_base_path
        if split_base_path is None:
            split_base_path = onet_base_path
        split_file = os.path.join(split_base_path, (split + '.lst'))
        if image_base_path is None:
            image_base_path = onet_base_path
        self.image_base_path = image_base_path
        
        with open(split_file, 'r') as f:
            models_c = f.read().split('\n')
        if '' in models_c:
            models_c.remove('')
        self.models = [{'category': m.split("/")[-2], 'model': m.split("/")[-1]} for m in models_c]
        # print(self.models)
        self.hparams = hparams
        
        # setup text condition
        if DatasetSpec.TEXT_EMBEDDING in self.spec:
            self.text_emb_path = text_emb_path
            self.null_text_emb = torch.load(null_embed_path)
            self.max_text_len = max_text_len
            self.text_embed_drop_prob = text_embed_drop_prob
        
        self.duplicate_num = duplicate_num

    def __len__(self):
        return len(self.models) * self.duplicate_num
            
    def get_name(self):
        return f"{self.custom_name}-{self.split}"

    def get_short_name(self):
        return self.custom_name
    
    def get_null_text_emb(self):
        null_text_emb = self.null_text_emb['text_embed_sd_model.last_hidden_state'] # 2, 1024
        return self.padding_text_emb(null_text_emb)
        
    def padding_text_emb(self, text_emb):
        padded_text_emb = torch.zeros(self.max_text_len, text_emb.shape[1])
        padded_text_emb[:text_emb.shape[0]] = text_emb
        mask = torch.zeros(self.max_text_len)
        mask[:text_emb.shape[0]] = 1
        return padded_text_emb, mask.bool()
        
    def _get_item(self, data_id, rng):
        category = self.models[data_id % len(self.models)]['category']
        model = self.models[data_id % len(self.models)]['model']
        data = {}
        input_data = torch.load(os.path.join(self.onet_base_path, category, model) + ".pkl", weights_only=False)
        input_points = input_data['points']
        input_normals = input_data['normals'].jdata
        if DatasetSpec.SHAPE_NAME in self.spec:
            data[DatasetSpec.SHAPE_NAME] = "/".join([category, model])

        if DatasetSpec.TARGET_NORMAL in self.spec:
            data[DatasetSpec.TARGET_NORMAL] = input_normals
    
        if DatasetSpec.INPUT_PC in self.spec:
            data[DatasetSpec.INPUT_PC] = input_points
                
        if DatasetSpec.GT_DENSE_PC in self.spec:
            data[DatasetSpec.GT_DENSE_PC] = input_points

        if DatasetSpec.GT_DENSE_NORMAL in self.spec:
            data[DatasetSpec.GT_DENSE_NORMAL] = input_normals

        if DatasetSpec.TEXT_EMBEDDING in self.spec:
            # first sample prob to drop text embedding
            if random.random() < self.text_embed_drop_prob:
                # drop the text
                text_emb, text_mask = self.get_null_text_emb()
                caption = ""
            else:
                text_emb_path = os.path.join(self.text_emb_path, model + ".pkl")
                if os.path.exists(text_emb_path):
                    text_emb_data = torch.load(text_emb_path)
                    text_emb = text_emb_data['text_embed_sd_model.last_hidden_state']
                    text_emb, text_mask = self.padding_text_emb(text_emb)
                    caption = text_emb_data['caption']
                else:
                    text_emb, text_mask = self.get_null_text_emb()
                    caption = ""
            data[DatasetSpec.TEXT_EMBEDDING] = text_emb.detach()
            data[DatasetSpec.TEXT_EMBEDDING_MASK] = text_mask.detach()
            data[DatasetSpec.TEXT] = caption
        # print(data)
        return data
