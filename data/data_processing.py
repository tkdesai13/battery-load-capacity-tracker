""" This module is used to process the data and split the data into training and validation set"""

import pandas as pd
import numpy as np

from typing import List, Tuple, Dict


class DataProcessor:
    def __init__(self, config_data: dict):
        self.config_data = config_data

    def load_data(self, data_path: str) -> pd.DataFrame:
        data = pd.read_csv(data_path)
        return data

    def preprocess_data(self, data: pd.DataFrame):
        data = data.dropna()
        return data
    
    def split_data(self, data: pd.DataFrame, train_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_data = data.sample(frac=train_size, random_state=42)
        val_data = data.drop(train_data.index)
        return train_data, val_data
    
    def process_data(self, data_path: str, train_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = self.load_data(data_path)
        data = self.preprocess_data(data)
        train_data, val_data = self.split_data(data, train_size)
        return train_data, val_data
