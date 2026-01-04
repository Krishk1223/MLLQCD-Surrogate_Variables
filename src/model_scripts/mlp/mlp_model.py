import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from ..base_model import BaseModel
from pathlib import Path
import json


class MLP(nn.Module, BaseModel):
    """Multi-Layer Perceptron for correlator prediction."""
    
    def __init__(self, config, experiment_folder, input_dim=None, output_dim=None):
        nn.Module.__init__(self)
        BaseModel.__init__(self, config)
        
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        
        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        # Paths
        project_root = Path(__file__).parent.parent.parent.parent
        self.data_path = project_root / config['data']['input_data_path'] / experiment_folder
        self.results_dir = project_root / 'results' / experiment_folder
        self.model_dir = project_root / 'models' / experiment_folder / 'mlp_model'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model dimensions (set later if not provided)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = None
        self.is_built = False
    
    def build_model(self, input_dim=None, output_dim=None):
        """Build the MLP network."""
        if input_dim is not None:
            self.input_dim = input_dim
        if output_dim is not None:
            self.output_dim = output_dim
        
        if self.input_dim is None or self.output_dim is None:
            raise ValueError("input_dim and output_dim must be set before building")
        
        hidden_dims = self.model_config.get('hidden_dims', [256, 128, 64])
        dropout = self.model_config.get('dropout', 0.1)
        
        layers = []
        prev_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        self.is_built = True
        self.is_trained = False
        return self
    
    def forward(self, x):
        return self.network(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def train_model(self, train_X, train_y, eval_X=None, eval_y=None):
        """
        Train the MLP.
        
        Args:
            train_X: Training features (N, input_dim) numpy array
            train_y: Training targets (N, output_dim) numpy array
            eval_X, eval_y: Optional evaluation data
        """
        self.to(self.device)
        
        # Create dataloaders
        batch_size = self.training_config.get('batch_size', 64)
        train_dataset = TensorDataset(
            torch.FloatTensor(train_X), 
            torch.FloatTensor(train_y)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        eval_loader = None
        if eval_X is not None and eval_y is not None:
            eval_dataset = TensorDataset(
                torch.FloatTensor(eval_X),
                torch.FloatTensor(eval_y)
            )
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer
        lr = self.training_config.get('learning_rate', 1e-3)
        weight_decay = self.training_config.get('weight_decay', 0.0)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Loss
        criterion = nn.MSELoss()
        
        # Tensorboard
        writer = SummaryWriter(self.results_dir / 'tensorboard_logs')
        
        # Training loop
        epochs = self.training_config.get('num_epochs', 100)
        best_eval_loss = float('inf')
        patience = self.training_config.get('early_stopping_patience', 15)
        patience_counter = 0
        best_state_dict = None
        
        print(f"Training MLP for {epochs} epochs on {self.device}")
        
        for epoch in range(epochs):
            # Train
            self.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                output = self(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            writer.add_scalar('Loss/train', train_loss, epoch)
            
            # Eval
            if eval_loader is not None:
                self.eval()
                eval_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in eval_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        output = self(X_batch)
                        eval_loss += criterion(output, y_batch).item()
                eval_loss /= len(eval_loader)
                writer.add_scalar('Loss/eval', eval_loss, epoch)
                
                # Early stopping - store best state in memory
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                    best_state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 20 == 0:
                msg = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}"
                if eval_loader:
                    msg += f", Eval Loss: {eval_loss:.6f}"
                print(msg)
        
        writer.close()
        self.is_trained = True
        
        # Save best model once at the end
        if best_state_dict is not None:
            best_model_path = self.model_dir / 'best_model.pth'
            torch.save(best_state_dict, best_model_path)
            print(f"Best model saved to: {best_model_path}")
            # Load best weights into model
            self.load_state_dict(best_state_dict)
        
        return self
    
    def predict_model(self, X):
        """Make predictions on numpy array X."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self(X_tensor).cpu().numpy()
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
            path = self.model_dir / 'mlp_model.pt'
        torch.save(self.state_dict(), path)
    
    def load_model(self, path=None):
        if path is None:
            path = self.model_dir / 'mlp_model.pt'
        self.load_state_dict(torch.load(path))
