from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

class BaseModel(ABC):
    def __init__(self,config):
        self.config = config
        self.Model = None
        self.is_trained = False

    @abstractmethod
    def build_model(self):
        pass

    def train(self, train_data, val_data):
        pass

    def bias_correction(self, bias_corr_data):
        pass

    def predict(self, test_data):
        pass

    def save_model(self, save_path):
        pass

    def load_model(self, load_path):
        pass

