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

    def compute_bias_correction(self):
        pass
    
    def apply_bias_correction(self, y_pred, bias_vector):
        """Applies bias correction to the predictions.
        Args:
            y_pred: Model predictions (numpy array)
            bias_vector: Bias correction vector (numpy array)
        Returns:
            C_corr = C_pred - bias_correction_vector.
        """
        return y_pred - bias_vector

    def predict(self, test_data):
        pass

    def save_model(self, save_path):
        pass

    def load_model(self, load_path):
        pass

