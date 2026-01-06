from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar


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

    def _find_optimal_boost_factor(self, y_pred, y_input, y_target, bounds=(0.5, 2.0)):
        """Find optimal boost factor minimising MSE to target using Brent's method."""
        pred_signs = np.sign(y_pred)
        abs_pred, abs_input = np.abs(y_pred), np.maximum(np.abs(y_input), 1e-30)

        def mse(b):
            boosted = (abs_input ** b) * (abs_pred / (abs_input ** b + 1e-30))
            return np.mean((boosted * pred_signs - y_target) ** 2)

        return minimize_scalar(mse, bounds=bounds, method='bounded').x

    def apply_boosted_ratio(self, y_pred, y_input, y_target=None, optimise=False, boost_factor=1.0):
        """Apply boosted ratio method: C_boosted = C_input^b * (C_pred / C_input^b).
        
        Args:
            y_pred: ML predictions (low precision)
            y_input: High precision inputs
            y_target: Targets (required if optimise=True)
            optimise: Find optimal boost factor
            boost_factor: Manual factor (ignored if optimise=True)
        
        Returns:
            (boosted_predictions, boost_factor_used)
        """
        if optimise:
            if y_target is None:
                raise ValueError("y_target required when optimise=True")
            boost_factor = self._find_optimal_boost_factor(y_pred, y_input, y_target)

        pred_signs = np.sign(y_pred)
        abs_pred, abs_input = np.abs(y_pred), np.maximum(np.abs(y_input), 1e-30)
        boosted = (abs_input ** boost_factor) * (abs_pred / (abs_input ** boost_factor + 1e-30))
        
        return boosted * pred_signs, boost_factor

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

