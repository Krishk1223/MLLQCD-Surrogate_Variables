from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path


class BaseModel(ABC):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.is_trained = False
        self._validate_correction_methods()

    def _validate_correction_methods(self):
        """Ensure bias correction and ratio boosting are mutually exclusive."""
        if self.config.get('bias_correction', False) and self.config.get('ratio_boosting', False):
            raise ValueError("bias_correction and ratio_boosting cannot both be True.")

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_model(self, train_data, val_data):
        pass

    @abstractmethod
    def predict_model(self, test_data):
        pass

    def compute_bias_correction(self):
        pass

    def apply_bias_correction(self, y_pred, bias_vector):
        """Apply bias correction: C_corr = C_pred - bias_vector."""
        return y_pred - bias_vector

    def apply_ratio_method(self, y_pred, y_input):
        """Apply ratio method: C_ratio = C_pred * (C_input / <C_input>).
        
        This transfers high-precision statistical fluctuations from the input
        to the ML predictions, improving the covariance structure.
        
        Args:
            y_pred: ML predictions (N, T)
            y_input: High precision inputs (N, T)
            y_target: Not used (kept for API compatibility)
            optimise: Not used (kept for API compatibility)
            boost_factor: Not used (kept for API compatibility)
        
        Returns:
            (ratio_predictions, 1.0)
        """
        # Mean over samples (axis=0), keeping timeslice dimension
        input_mean = np.mean(y_input, axis=0, keepdims=True)
        
        # Ratio method: scale predictions by input/mean(input)
        # This transfers the per-sample fluctuations to the predictions
        ratio = y_input / (input_mean + 1e-30)
        y_ratio = y_pred * ratio
        
        return y_ratio, 1.0

    def save_ratio_predictions(self, y_boosted, boost_factor, save_path, method='ratio'):
        """Save ratio-boosted predictions to subfolder."""
        ratio_path = Path(save_path) / f"{method}_predictions"
        ratio_path.mkdir(parents=True, exist_ok=True)
        np.save(ratio_path / "correlator_predictions.npy", y_boosted)
        np.save(ratio_path / "boost_factor.npy", np.array([boost_factor]))
        return ratio_path

    def save_model(self, save_path):
        pass

    def load_model(self, load_path):
        pass

