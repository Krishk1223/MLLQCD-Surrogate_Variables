import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from ..base_model import BaseModel
from pathlib import Path
import joblib
import json


class GBR(BaseModel):
    """Gradient Boosted Regressor for correlator prediction.
    
    Trains one GBR per output timeslice since sklearn GBR only supports 
    single-output regression.
    """
    
    def __init__(self, config, experiment_folder):
        super().__init__(config)
        
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        
        # Paths
        project_root = Path(__file__).parent.parent.parent.parent
        self.data_path = project_root / config['data']['input_data_path'] / experiment_folder
        self.results_dir = project_root / 'results' / experiment_folder
        self.model_dir = project_root / 'models' / experiment_folder / 'gbr_model'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.models = []
        self.output_dim = None
        self.is_built = False
    
    def build_model(self, output_dim=None):
        """Initialize model parameters."""
        self.output_dim = output_dim
        self.models = []
        self.is_built = True
        self.is_trained = False
        return self
    
    def count_parameters(self):
        """Return number of trees * output_dim as proxy for complexity."""
        n_estimators = self.model_config.get('n_estimators', 100)
        return n_estimators * (self.output_dim or 1)
    
    def train_model(self, train_X, train_y):
        """
        Train GBR models (one per output timeslice).
        
        Args:
            train_X: Training features (N, input_dim) numpy array
            train_y: Training targets (N, output_dim) numpy array
        """
        n_estimators = self.model_config.get('n_estimators', 100)
        max_depth = self.model_config.get('max_depth', 5)
        learning_rate = self.model_config.get('learning_rate', 0.1)
        
        self.output_dim = train_y.shape[1]
        self.models = []
        
        print(f"Training GBR ({n_estimators} trees, depth={max_depth}) for {self.output_dim} timeslices...")
        
        for t in range(self.output_dim):
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )
            model.fit(train_X, train_y[:, t])
            self.models.append(model)
            
            if (t + 1) % 20 == 0:
                print(f"  Completed {t+1}/{self.output_dim} timeslices")
        
        self.is_trained = True
        return self
    
    def predict_model(self, X):
        """Make predictions on numpy array X."""
        predictions = np.zeros((X.shape[0], self.output_dim))
        for t, model in enumerate(self.models):
            predictions[:, t] = model.predict(X)
        return predictions
    
    def compute_bias_correction(self, bias_X, bias_y, target_scaler):
        """Compute bias correction vector."""
        predictions = self.predict_model(bias_X)
        predictions_unscaled = target_scaler.inverse_transform(predictions)
        targets_unscaled = target_scaler.inverse_transform(bias_y)
        bias_vector = np.mean(targets_unscaled - predictions_unscaled, axis=0)
        print(f"Bias correction: mean abs = {np.mean(np.abs(bias_vector)):.6f}")
        return bias_vector
    
    def evaluate_model(self, test_X, test_y, target_scaler, bias_vector=None, save_predictions=True):
        """Evaluate model on test set."""
        predictions_scaled = self.predict_model(test_X)
        predictions = target_scaler.inverse_transform(predictions_scaled)
        targets = target_scaler.inverse_transform(test_y)
        
        if bias_vector is not None:
            predictions = predictions + bias_vector
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        metrics = {'mse': float(mse), 'mae': float(mae)}
        print(f"Test MSE: {mse:.6f}, MAE: {mae:.6f}")
        
        if save_predictions:
            np.save(self.results_dir / 'correlator_predictions.npy', predictions)
            np.save(self.results_dir / 'test_targets.npy', targets)
            with open(self.results_dir / 'scaled_evaluation_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        
        return metrics, predictions
    
    def save_model(self, path=None):
        if path is None:
            path = self.model_dir / 'gbr_models.joblib'
        joblib.dump(self.models, path)
        print(f"Models saved to {path}")
    
    def load_model(self, path=None):
        if path is None:
            path = self.model_dir / 'gbr_models.joblib'
        self.models = joblib.load(path)
        self.output_dim = len(self.models)
        self.is_trained = True
