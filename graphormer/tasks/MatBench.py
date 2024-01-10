# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Sequence, Union

import pickle
from functools import lru_cache

import lmdb

import numpy as np
import torch
from torch import Tensor
from fairseq.data import (
    FairseqDataset,
    BaseWrapperDataset,
    NestedDictionaryDataset,
    data_utils,
    NumSamplesDataset,
)
from ..data.dataset import (
    BatchedDataDataset,
    TargetDataset,
    GraphormerDataset,
    EpochShuffleDataset,
)
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from dataclasses import dataclass, field
from omegaconf import II

from ..data.dataset import EpochShuffleDataset

from sklearn.model_selection import train_test_split
import pandas as pd

from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch

from matbench.bench import MatbenchBenchmark

def pad_1d(samples: Sequence[Tensor], fill=0, multiplier=8):
    max_len = max(x.size(0) for x in samples)
    max_len = (max_len + multiplier - 1) // multiplier * multiplier
    n_samples = len(samples)
    out = torch.full(
        (n_samples, max_len, *samples[0].shape[1:]), fill, dtype=samples[0].dtype
    )
    for i in range(n_samples):
        x_len = samples[i].size(0)
        out[i][:x_len] = samples[i]
    return out


class CrystDataset(Dataset):
    def __init__(self,data,pad_fill=0):
        super().__init__()
        self.data = data

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        data_dict = self.data[index]

        graph_dict = data_dict['graph_arrays']

        data = Data(
            frac_coords=torch.Tensor(graph_dict["frac_coords"]),
            atom_types=torch.LongTensor(graph_dict["atom_types"]),
            lengths=torch.Tensor(graph_dict["lengths"]).view(1, -1),
            angles=torch.Tensor(graph_dict["angles"]).view(1, -1),
            num_atoms=graph_dict["num_atoms"],
            num_nodes=graph_dict["num_atoms"],  # special attribute used for batching in pytorch geometric
            y=torch.Tensor([data_dict["property"]]).view(1, -1),
        )

        return data

    def __len__(self):
        return len(self.data)

    def collater(self, samples):
        batch = Batch.from_data_list(samples)
        return batch

@dataclass
class MatBenchConfig(FairseqDataclass):
    seed: int = II("common.seed")

    MatBench_name: str = field(
        default="",
        metadata={"help": "name of the MatBench dataset"},
    )

def build_crystal_graph(crystal,scale_length=False):
    """
    """
    graph_arrays = {}

    graph_arrays["frac_coords"] = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    graph_arrays["lattice_matrix"] = crystal.lattice.matrix
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    graph_arrays["atom_types"] = np.array(atom_types)
    graph_arrays["lengths"], graph_arrays["angles"] = np.array(lengths), np.array(angles)
    graph_arrays["num_atoms"] = graph_arrays["atom_types"].shape[0]

    if scale_length:
        graph_arrays["scale_length"] = lengths / float(graph_arrays["num_atoms"])**(1/3)

    return graph_arrays


def preprocess(data:pd.DataFrame,num_workers):

    def process_one(row):
        material_id = row.name
        crystal = row['structure']
        property = row[-1]
        graph_arrays = build_crystal_graph(crystal)
        result_dict = {
            'mp_id': material_id,
            'graph_arrays': graph_arrays,
            'property': property,
        }
        return result_dict
    
    result = data.apply(process_one, axis=1)
    result = list(result)

    return result


@register_task("MatBench",dataclass=MatBenchConfig)
class MatBenchTask(FairseqTask):
    """
    Task for training Graphormer on MatBench.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        subset = [cfg.MatBench_name]
        mb = MatbenchBenchmark(autoload=False, subset=subset)
        task = mb.tasks
        for task in mb.tasks:
            task.load()
            fold = 0
            train_val_inputs,train_val_outputs = task.get_train_and_val_data(fold)
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

        train_val_data = pd.concat([train_val_inputs.to_frame(),train_val_outputs.to_frame()],axis=1)
        test_data = pd.concat([test_inputs.to_frame(),test_outputs.to_frame()],axis=1) 
        train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=cfg.seed)

        train_data = preprocess(train_data,0)
        val_data = preprocess(val_data,0)
        test_data = preprocess(test_data,0)

        print(f" > Loading MatBench dataset {cfg.MatBench_name} ...")

        self.train_dataset = CrystDataset(train_data)
        self.valid_dataset = CrystDataset(val_data)
        self.test_dataset = CrystDataset(test_data)


    @property
    def target_dictionary(self):
        return None

    def load_dataset(self, split, combine=False, **kwargs):
        assert split in ["train", "valid", "test"]

        if split == "train":
            crystal_data = self.train_dataset
        elif split == "valid":
            crystal_data = self.valid_dataset
        elif split == "test":
            crystal_data = self.test_dataset

        print(" > Loading {} ...".format(split))

        target = TargetDataset(crystal_data)

        dataset = NestedDictionaryDataset(
            {
                "nsamples": NumSamplesDataset(),
                "net_input": {
                    "data":crystal_data
                },
                "targets": target,
            },
            sizes=[np.zeros(len(crystal_data))],
        )

        if split == "train":
            dataset = EpochShuffleDataset(
                dataset,
                num_samples=len(crystal_data),
                seed=self.cfg.seed,
            )

        print("| Loaded {} with {} samples".format(split, len(dataset)))
        self.datasets[split] = dataset
        return self.datasets[split]
