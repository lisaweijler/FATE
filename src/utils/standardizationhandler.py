from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from src.utils.json_file_handler import JsonFileLoader, JsonFileWriter


class StandardizationHandler():

    def __init__(self, preload_dir: Path, dataset_type: str, markers: List[str]) -> None:
        self.preload_dir = Path(preload_dir)
        self.dataset_type = dataset_type
        self.markers = markers

    def prepare_stats(self, filepaths: List[str]):
        """
        either loads already generated stats from file or retrieves the event stats of each sample and saves them to file
        """
        stats_dir = self.preload_dir.parent if self.preload_dir.name in ["train", "eval", "test"] else self.preload_dir
        stats_filepath = stats_dir / Path("stats.json")
        if self.dataset_type == "train":
            mean_list = []
            std_list = []
            for filepath in tqdm(filepaths):
                path = self.preload_dir / Path(Path(filepath).with_suffix("").name + ".pkl")
                events = pd.read_pickle(path)
                mean_list.append(events.loc[:, self.markers].mean())
                std_list.append(events.loc[:, self.markers].std())

            self.stats = {"mean": pd.concat(mean_list, axis = 1).transpose().mean(), "std": pd.concat(std_list, axis = 1).transpose().mean()}
            fileWriter = JsonFileWriter(stats_filepath)
            fileWriter.save_as_json(self.stats)
        else:
            assert stats_filepath.exists() and stats_filepath.is_file(), f"stats_filepath '{stats_filepath}' does not exists or is not a file"
            fileReader = JsonFileLoader(stats_filepath)
            self.stats = fileReader.loadJsonFile()
            for key in self.stats.keys():
                self.stats[key] = pd.read_json(self.stats[key], typ= "series")
        
        self.mean_tensor = torch.Tensor(self.stats["mean"])
        self.std_tensor = torch.Tensor(self.stats["std"])
    
    def standardize(self, events: np.ndarray) -> pd.DataFrame:
        """
        standardizes the given events as pd.DataFrame / np.ndarray
        """
        return (events - self.stats["mean"]) / self.stats["std"]

    def standardize_tensor(self, events: torch.Tensor) -> torch.Tensor:
        """
        standardizes the given events as tensors
        """
        return (events - self.mean_tensor) / self.std_tensor
