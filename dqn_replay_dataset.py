import os
import gzip
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils.pytorch_utils import *


class DQNReplayDataset(Dataset):
    """
    A dataset of observations from a one checkpoint of one game.
    It saves a tensor of dimension: (dataset_size, h, w)
    and given an index i returns a slice starting at i and
    ending in i plus a number of frames: (slice_size, h, w).
    The slice size should be equivalent to the number of frames stacked
    during the RL phase.
    In add adjacent mode the dataset returns three stacked observations
    the observation before i the observation i and the observation after i.
    (3, slice_size, h, w)
    """

    def __init__(
        self,
        data_path: Path,
        game: str,
        checkpoint: int,
        frames: int,
        max_size: int,
        transform,
        actions=False,
        start_index=0,
    ) -> None:
        self.actions = None
        data = torch.tensor([])
        self.start_index = start_index

        filename = Path(data_path / f"{game}/observation_{checkpoint}.gz")
        print(f"Loading {filename}")

        zipFile = gzip.GzipFile(filename=filename)
        loaded_data = np.load(zipFile)
        loaded_data_capped = np.copy(
            loaded_data[self.start_index : self.start_index + max_size]
        )

        print(f"Using {loaded_data.size * loaded_data.itemsize} bytes")
        print(f"Shape {loaded_data.shape}")

        data = torch.from_numpy(loaded_data_capped)
        setattr(self, "observation", data)

        del loaded_data
        del zipFile
        del loaded_data_capped

        if actions:
            actions_filename = Path(data_path / f"{game}/action_{checkpoint}.gz")
            actions_zipFile = gzip.GzipFile(filename=actions_filename)
            actions_loaded_data = np.load(actions_zipFile)
            actions_data_capped = np.copy(
                actions_loaded_data[self.start_index : self.start_index + max_size]
            )
            data = torch.from_numpy(actions_data_capped)
            setattr(self, "actions", data)

        self.size = min(data.shape[0], max_size)
        self.game = game
        self.frames = frames
        self.effective_size = self.size - self.frames + 1
        self.transform = transform


    def __len__(self):
        return self.effective_size

    def __getitem__(self, index: int) -> torch.Tensor:
        time_ind = index % self.effective_size

        res_action = self.actions[time_ind]

        curr_obs = None

        if self.frames <= 1:
            curr_obs = self.observation[time_ind]
        else:
            curr_slice = slice(time_ind, time_ind + self.frames)
            curr_obs = self.observation[curr_slice]

        return self.transform(curr_obs), res_action


class MultiDQNReplayDataset(Dataset):
    """
    This dataset corresponds to the concatenation of several DQNReplayDataset.
    Meaning that it contains several checkpoints from several games.
    """

    def __init__(
        self,
        data_path: Path,
        games: Union[List[str], str],
        checkpoints: List[int],
        frames: int,
        max_size: int,
        transform,
        actions=True,
        start_index=0,
    ) -> None:
        self.actions = actions
        self.n_checkpoints_per_game = len(checkpoints)

        if isinstance(games, str):
            games = [games]

        self.datasets = [
            DQNReplayDataset(
                data_path,
                game,
                ckpt,
                frames,
                max_size,
                transform,
                actions,
                start_index,
            )
            for ckpt in checkpoints
            for game in games
        ]

        self.n_datasets = len(self.datasets)
        self.single_dataset_size = len(self.datasets[0])

    def __len__(self) -> int:
        return self.n_datasets * self.single_dataset_size

    def __getitem__(self, index: int) -> torch.Tensor:
        multidataset_index = index % self.n_datasets
        dataset_index = index // self.n_datasets
        res_obs, res_action = self.datasets[multidataset_index][dataset_index]
        return [res_obs, res_action]

