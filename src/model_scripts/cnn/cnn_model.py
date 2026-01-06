import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from ..base_model import BaseModel
from pathlib import Path
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import json
from typing import Optional


class CNN(nn.Module, BaseModel):
    def __init__(self, config, experiment_folder, input_channels=1, output_channels=None):
        """
        1D Convolutional Neural Network for sequence-to-sequence regression.
        
        Args:
            config: Configuration dictionary
            experiment_folder: Name of experiment folder
            input_channels: Number of input channels/sequences (default 1)
            output_channels: Number of output channels (default: same as input_channels)
        """
        nn.Module.__init__(self)
        BaseModel.__init__(self, config)

        # Config
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        self.logging_config = config['logging']
        self.device = None
        
        # Output channels default to same as input (for parity: 2 in -> 2 out)
        if output_channels is None:
            output_channels = input_channels

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # Project root and paths
        project_root = Path(__file__).parent.parent.parent.parent
        self.data_path = project_root / self.config['data']['input_data_path'] / experiment_folder
        self.results_dir = project_root / 'results' / experiment_folder / 'cnn'
        self.model_dir = project_root / 'models' / experiment_folder / 'cnn_model'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # CNN architecture parameters
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = self.model_config.get('hidden_channels', [64, 128, 64])
        self.kernel_sizes = self.model_config.get('kernel_sizes', [5, 3, 3])
        self.dropout_rate = self.model_config.get('dropout', 0.2)
        self.use_batch_norm = self.model_config.get('use_batch_norm', True)
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(self.dropout_rate)
        
        in_channels = self.input_channels
        for i, (out_ch, kernel_size) in enumerate(zip(self.hidden_channels, self.kernel_sizes)):
            # Padding to maintain sequence length (same padding)
            padding = kernel_size // 2
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_ch, kernel_size, padding=padding)
            )
            if self.use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(out_ch))
            in_channels = out_ch
        
        # Activation
        self.activation = nn.LeakyReLU()
        
        # Output projection: map from hidden_channels[-1] to output_channels
        self.output_conv = nn.Conv1d(self.hidden_channels[-1], self.output_channels, kernel_size=1)
        
        self.is_built = True

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_sequences)
               from dataloader standard format
               
        Returns:
            Output tensor of shape (batch, seq_len, 1)
        """
        # Permute to Conv1d format
        x = x.permute(0, 2, 1)
        
        # Pass through convolutional layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output projection
        x = self.output_conv(x)
        
        # Permute back to standard format
        x = x.permute(0, 2, 1)
        
        return x

    def count_parameters(self):
        """Count trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def build_model(self):
        """Verify model is built and ready for training."""
        if not self.is_built:
            raise ValueError("Model components are not built properly.")
        self.is_trained = False
        return self

    def train_model(self, trainloader, evalloader):
        """Training loop with TensorBoard logging, early stopping, and LR scheduling."""
        self.to(self.device)

        # Optimizer
        if self.training_config['optimiser'] == 'adamw':
            optimiser = torch.optim.AdamW(
                self.parameters(),
                lr=self.training_config['learning_rate'],
                weight_decay=self.training_config['weight_decay']
            )
        else:
            optimiser = torch.optim.Adam(
                self.parameters(),
                lr=self.training_config['learning_rate'],
                weight_decay=self.training_config['weight_decay']
            )

        # Loss function - use reduction='none' for timeslice weighting
        if self.training_config['loss_function'] == 'HuberLoss':
            criterion = nn.HuberLoss(reduction='none')
        elif self.training_config['loss_function'] == 'MAE':
            criterion = nn.L1Loss(reduction='none')
        else:
            criterion = nn.MSELoss(reduction='none')

        # Get sequence length from first batch
        sample_batch = next(iter(trainloader))
        seq_len = sample_batch['features'].shape[1]
        
        # Create timeslice weights: boost importance of t=10-40 where SNR is low but non-zero
        t = torch.arange(seq_len, dtype=torch.float32, device=self.device)
        time_weights = 1.0 + 5.0 * torch.exp(-((t - 20) ** 2) / (2 * 15 ** 2))
        time_weights = time_weights / time_weights.mean()
        self.time_weights = time_weights.view(1, -1, 1)

        # LR scheduler
        scheduler = None
        if self.training_config['scheduler']['use_scheduler']:
            if self.training_config['scheduler']['type'] == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimiser,
                    patience=self.training_config['scheduler']['patience'],
                    factor=self.training_config['scheduler']['factor'],
                    mode=self.training_config['scheduler']['mode']
                )
            elif self.training_config['scheduler']['type'] == 'StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimiser,
                    step_size=self.training_config['scheduler']['step_size'],
                    gamma=self.training_config['scheduler']['gamma']
                )

        # Gradient clipping
        max_grad_norm = None
        if self.training_config['gradient_clipping']['use_clipping']:
            max_grad_norm = self.training_config['gradient_clipping']['max_norm']

        # Warmup scheduler
        warmup_epochs = self.training_config['scheduler'].get('warmup_epochs', 0)
        warmup_scheduler = None
        if warmup_epochs > 0:
            def warmup_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                return 1.0
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=warmup_lambda)
            print(f"Using LR warmup for {warmup_epochs} epochs")

        # Training tracking
        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None

        # TensorBoard
        writer_dir = self.results_dir / 'tensorboard_logs'
        writer_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(writer_dir))

        writer.add_text('Hyperparameters', str({
            'hidden_channels': self.hidden_channels,
            'kernel_sizes': self.kernel_sizes,
            'dropout': self.dropout_rate,
            'learning_rate': self.training_config['learning_rate'],
            'batch_size': self.training_config['batch_size']
        }))

        train_loader = trainloader
        eval_loader = evalloader

        # Training loop
        for epoch in range(int(self.training_config['num_epochs'])):
            self.train()
            total_train_loss = 0.0

            for batch_idx, batch_data in enumerate(train_loader):
                inputs = batch_data['features'].to(self.device)
                targets = batch_data['targets'].to(self.device)

                optimiser.zero_grad()
                predictions = self.forward(inputs)
                
                # Apply timeslice weights
                raw_loss = criterion(predictions, targets)
                seq_len = predictions.shape[1]
                weights = self.time_weights[:, :seq_len, :]
                value_loss = (raw_loss * weights).mean()
                
                pred_derivative = predictions[:, 1:, :] - predictions[:, :-1, :]
                target_derivative = targets[:, 1:, :] - targets[:, :-1, :]
                slope_loss = criterion(pred_derivative, target_derivative).mean()
                loss = value_loss + 2.0 * slope_loss

                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimiser.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            self.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch_data in eval_loader:
                    inputs = batch_data['features'].to(self.device)
                    targets = batch_data['targets'].to(self.device)
                    predictions = self.forward(inputs)
                    seq_len = predictions.shape[1]
                    weights = self.time_weights[:, :seq_len, :]
                    loss = (criterion(predictions, targets) * weights).mean()
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(eval_loader)

            # Scheduler step
            if warmup_scheduler is not None and epoch < warmup_epochs:
                warmup_scheduler.step()
            elif scheduler is not None:
                if self.training_config['scheduler']['type'] == 'ReduceLROnPlateau':
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            current_lr = optimiser.param_groups[0]['lr']

            # Logging
            writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
            writer.add_scalar('learning_rate', current_lr, epoch + 1)
            writer.add_scalars('Loss/Train_vs_Validation', {
                'Train': avg_train_loss,
                'Validation': avg_val_loss
            }, epoch + 1)

            print(f"Epoch [{epoch + 1}/{self.training_config['num_epochs']}], "
                  f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            # Early stopping and checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Store best state dict in memory (save only at end)
                best_state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.training_config['early_stopping_patience']:
                    print("Early stopping triggered.")
                    writer.add_text('Training Stopped', f"Early stopping at epoch {epoch + 1}")
                    break

        # Final logging
        writer.add_hparams({
            'hidden_channels': str(self.hidden_channels),
            'kernel_sizes': str(self.kernel_sizes),
            'lr': self.training_config['learning_rate'],
        }, {
            'best_val_loss': best_val_loss
        })
        writer.close()

        # Save best model (once at the end)
        if self.logging_config['save_best_model'] and best_state_dict is not None:
            best_model_path = self.model_dir / 'best_model.pth'
            torch.save(best_state_dict, best_model_path)
            print(f"Best model saved to: {best_model_path}")

        # Save final model (current state, not best)
        if self.logging_config.get('save_final_model', False):
            final_model_path = self.model_dir / 'final_model.pth'
            self.save_model(final_model_path, override=True)
            print(f"Final model saved to: {final_model_path}")

        self.is_trained = True
        print(f"Training completed.")
        print(f"TensorBoard logs saved to: {writer_dir}")
    
    def compute_bias_correction(self, biasloader, target_scaler: StandardScaler,
                                   target_signs: Optional[np.ndarray] = None):
        """Compute bias correction vector from bias correction dataset.
        
        Args:
            biasloader: DataLoader for bias correction dataset
            target_scaler: Fitted StandardScaler for inverse transform
            target_signs: Signs array for inverse log transform (exp(x) * sign)
            
        Returns:
            bias_vector: Array of shape (seq_len,) with bias per time step in original space
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before computing bias correction.")
        
        # Check if bias vector already exists
        bias_path = self.model_dir / 'bias_correction_vector.npy'
        if bias_path.exists():
            print(f"Loading existing bias correction vector from: {bias_path}")
            return np.load(bias_path)
        
        self.to(self.device)
        bias_loader = biasloader
        predictions = []
        targets = []
        self.eval()
        
        with torch.no_grad():
            for batch_data in bias_loader:
                inputs = batch_data['features'].to(self.device)
                target = batch_data['targets'].to(self.device)
                output = self.forward(inputs)
                predictions.append(output.cpu())
                targets.append(target.cpu())
        
        predictions = torch.cat(predictions, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        
        # Squeeze to (N_configs, seq_len)
        predictions = predictions.squeeze(-1)
        targets = targets.squeeze(-1)
        
        # Inverse scaling (StandardScaler) to get log-transformed values
        n_configs, seq_len = predictions.shape
        predictions_unscaled = target_scaler.inverse_transform(predictions.reshape(-1, 1))
        predictions_unscaled = predictions_unscaled.reshape(n_configs, seq_len)
        targets_unscaled = target_scaler.inverse_transform(targets.reshape(-1, 1))
        targets_unscaled = targets_unscaled.reshape(n_configs, seq_len)
        
        # Apply inverse log transform: exp(x) * sign to get original correlator values
        if target_signs is not None:
            predictions_unscaled = np.exp(predictions_unscaled) * target_signs
            targets_unscaled = np.exp(targets_unscaled) * target_signs
        
        # Compute bias: average residual across all configs (now in original correlator space)
        bias_vector = np.mean(predictions_unscaled - targets_unscaled, axis=0)  # Shape: (seq_len,)
        
        # Save bias correction vector
        np.save(bias_path, bias_vector)
        print(f"Bias correction vector computed and saved to: {bias_path}")
        print(f"Bias vector shape: {bias_vector.shape}")
        print(f"Bias vector stats - Mean: {np.mean(bias_vector):.6f}, Std: {np.std(bias_vector):.6f}")
        
        return bias_vector
    
    def predict_model(self, testloader, target_scaler: Optional[StandardScaler] = None, 
                      bias_vector: Optional[np.ndarray] = None, save_predictions=True):
        """Generate predictions on test data.
        
        Args:
            testloader: DataLoader for test data
            target_scaler: Fitted StandardScaler for inverse transform (required if bias_vector provided)
            bias_vector: Pre-computed bias correction vector of shape (seq_len,). If provided,
                         predictions will be bias-corrected. Requires target_scaler.
            save_predictions: Whether to save predictions to disk
            
        Returns:
            all_predictions: Array of shape (N_configs, seq_len)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        if bias_vector is not None and target_scaler is None:
            raise ValueError("target_scaler is required when applying bias correction.")

        test_loader = testloader
        all_predictions = []
        self.eval()

        with torch.no_grad():
            for batch_data in test_loader:
                inputs = batch_data['features'].to(self.device)
                predictions = self.forward(inputs)
                all_predictions.append(predictions.cpu())

        all_predictions = torch.cat(all_predictions, dim=0).numpy()

        # Inverse scaling if scaler provided
        unscaled = False
        if isinstance(target_scaler, StandardScaler):
            unscaled = True
            original_shape = all_predictions.shape
            all_predictions = target_scaler.inverse_transform(all_predictions.reshape(-1, 1))
            all_predictions = all_predictions.reshape(original_shape)

        # Squeeze to (N_configs, seq_len)
        all_predictions = all_predictions.squeeze(-1)

        # Apply bias correction if provided
        bias_corrected = False
        if bias_vector is not None:
            all_predictions = all_predictions - bias_vector
            bias_corrected = True

        if save_predictions:
            if bias_corrected:
                np.save(self.results_dir / 'correlator_predictions.npy', all_predictions)
                print(f"Bias-corrected predictions saved with shape {all_predictions.shape} to: {self.results_dir / 'correlator_predictions.npy'}")
            elif unscaled:
                np.save(self.results_dir / 'correlator_predictions.npy', all_predictions)
                print(f"Test correlator predictions saved with shape {all_predictions.shape} to: {self.results_dir / 'correlator_predictions.npy'}")
            else:
                np.save(self.results_dir / 'scaled_predictions.npy', all_predictions)
                print(f"Test SCALED predictions saved with shape {all_predictions.shape} to: {self.results_dir / 'scaled_predictions.npy'}")

        return all_predictions

    def evaluate_model(self, testloader, target_scaler: Optional[StandardScaler] = None, 
                        bias_vector: Optional[np.ndarray] = None, save_predictions=True,
                        target_signs: Optional[np.ndarray] = None):
        """Evaluate model on test data and compute metrics.
        
        Args:
            testloader: DataLoader for test data
            target_scaler: Fitted StandardScaler for inverse transform (required if bias_vector provided)
            bias_vector: Pre-computed bias correction vector of shape (seq_len,). If provided,
                         predictions will be bias-corrected before computing metrics.
            save_predictions: Whether to save predictions to disk
            target_signs: Signs array for inverse log transform (exp(x) * sign)
            
        Returns:
            If unscaled: (scaled_metrics, mean_relative_error, predictions, relative_errors)
            If scaled only: (scaled_metrics, predictions)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")
        if bias_vector is not None and target_scaler is None:
            raise ValueError("target_scaler is required when applying bias correction.")

        predictions = []
        targets = []
        self.eval()
        test_loader = testloader

        with torch.no_grad():
            for batch_data in test_loader:
                inputs = batch_data['features'].to(self.device)
                target = batch_data['targets'].to(self.device)
                output = self.forward(inputs)
                predictions.append(output.cpu())
                targets.append(target.cpu())

        predictions = torch.cat(predictions, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

        # Squeeze to (N_configs, seq_len)
        predictions = predictions.squeeze(-1)
        targets = targets.squeeze(-1)

        # Scaled metrics (before any transforms)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)

        scaled_metrics = {
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'Mean Absolute Error': mae
        }

        print(f"\nEvaluation Metrics (Scaled):")
        for metric_name, metric_value in scaled_metrics.items():
            print(f"{metric_name}: {metric_value:.6f}")

        # Save scaled metrics
        metrics_path = self.results_dir / 'scaled_evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(scaled_metrics, f, indent=4)
        print(f"Evaluation SCALED metrics saved to: {metrics_path}")

        unscaled = False
        bias_corrected = False
        
        if isinstance(target_scaler, StandardScaler):
            unscaled = True
            n_configs, seq_len = predictions.shape
            predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1))
            predictions = predictions.reshape(n_configs, seq_len)
            targets = target_scaler.inverse_transform(targets.reshape(-1, 1))
            targets = targets.reshape(n_configs, seq_len)

            # Apply inverse log transform: exp(x) * sign
            if target_signs is not None:
                predictions = np.exp(predictions) * target_signs
                targets = np.exp(targets) * target_signs

            # Apply bias correction if provided
            if bias_vector is not None:
                predictions = predictions - bias_vector
                bias_corrected = True
                print(f"\nApplied bias correction to predictions.")

            # Relative error
            relative_errors = np.abs((targets - predictions) / (targets + 1e-10))
            mean_relative_error = np.mean(relative_errors)
            
            print(f"\nUnscaled Metrics{' (Bias-Corrected)' if bias_corrected else ''}:")
            print(f"Mean Relative Error: {mean_relative_error:.6f}")

        if save_predictions:
            if unscaled:
                np.save(self.results_dir / 'correlator_predictions.npy', predictions)
                print(f"Test correlator predictions{' (bias-corrected)' if bias_corrected else ''} with shape {predictions.shape} saved to: {self.results_dir / 'correlator_predictions.npy'}")
                np.save(self.results_dir / 'test_targets.npy', targets)
                print(f"Test targets with shape {targets.shape} saved to: {self.results_dir / 'test_targets.npy'}")
                np.save(self.results_dir / 'relative_errors.npy', relative_errors)
                print(f"Relative errors saved to: {self.results_dir / 'relative_errors.npy'}")
                return scaled_metrics, mean_relative_error, predictions, relative_errors
            else:
                np.save(self.results_dir / 'scaled_predictions.npy', predictions)
                print(f"Test SCALED predictions with shape {predictions.shape} saved to: {self.results_dir / 'scaled_predictions.npy'}")
                return scaled_metrics, predictions

    def save_model(self, save_path=None, override=False):
        """Save model state dict."""
        if not self.is_trained and not override:
            raise ValueError("Model must be trained before saving.")
        elif not self.logging_config['save_model']:
            raise ValueError("Model saving is disabled in configuration.")

        if save_path is None:
            save_path = self.model_dir / 'model.pth'
        else:
            save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path=None):
        """Load model state dict."""
        if load_path is None:
            load_path = self.model_dir / 'best_model.pth'
        else:
            load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        self.load_state_dict(torch.load(load_path, map_location=self.device))
        self.is_trained = True
        print(f"Model loaded from {load_path}")
