import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from ..base_model import BaseModel
from .transformer_components.InputEmbedding import InputEmbedding
from .transformer_components.PositionalEncoding import PositionalEncoding
from .transformer_components.EncoderLayer import EncoderLayer
from .transformer_components.RegressionHead import RegressionHead
from pathlib import Path
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import json
from typing import Optional


class Transformer(nn.Module, BaseModel):
    def __init__(self, config, experiment_folder, input_dim=1, output_dim=None, num_layers=None):
        nn.Module.__init__(self)
        BaseModel.__init__(self, config)

        #config:
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        self.logging_config = config['logging'] #will set up logging later
        self.device = None

        #Device setup:
        if torch.cuda.is_available():
            self.device = torch.device('cuda') #use CUDA if available
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps') # use metal performance shaders for macs with apple silicon
        else:
            self.device = torch.device('cpu') #cpu if nothing else available

        #project root and data path:
        project_root = Path(__file__).parent.parent.parent.parent 
        self.data_path = project_root / self.config['data']['input_data_path'] / experiment_folder
        self.results_dir = project_root / 'results' / experiment_folder / 'transformer'
        self.model_dir = project_root / 'models' / experiment_folder / 'transformer_model'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Determine number of layers
        if num_layers is None:
            num_layers = self.model_config.get('num_layers', 2)
            
        # Determine output_dim: use parameter if provided, else config default
        if output_dim is None:
            output_dim = self.model_config['regression_head']['output_dim']

        #Component initialisation
        self.input_embed = InputEmbedding(
            input_dim,
            self.model_config['d_model']
            )
        self.pos_encoder = PositionalEncoding(
            self.model_config['d_model'],
            self.model_config['max_len'],
            learnable = self.model_config['learnable_pos_encoding'] 
            )
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                self.model_config['d_model'],
                self.model_config['num_heads'],
                self.model_config['d_ff'],
                self.model_config['dropout'])
            for _ in range(num_layers)
        ])
        self.regressor_head = RegressionHead(
            self.model_config['d_model'],
            output_dim)

        self.is_built = True

    def forward(self, x):
        x = self.input_embed(x)
        x = self.pos_encoder(x)

        for layer in self.encoder_layers:
            x = layer(x)

        output = self.regressor_head(x)
        return output

    def count_parameters(self):
        """Count trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 
    
    def build_model(self):
        #Components built in init but method used to verify if trained
        if not self.is_built:
            raise ValueError("Model components are not built properly.")
        self.is_trained = False
        return self
    
    def train_model(self, trainloader, evalloader):
        #Training logic will be implemented here
        self.to(self.device)

        #Optimiser:
        if self.training_config['optimiser'] == 'adamw': #most common for transformers
            optimiser = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay = self.training_config['weight_decay']
            )

        # once there is some implenmentation in pytorch for Lion optimiser I will add it here.

        else: #default to adam
            optimiser = torch.optim.Adam(
                self.parameters(),
                lr=self.training_config['learning_rate'],
                weight_decay = self.training_config['weight_decay']
                )

        #Loss function:
        if self.training_config['loss_function'] == 'HuberLoss':
            criterion = nn.HuberLoss(reduction='none')
        elif self.training_config['loss_function'] == 'MAE':
            criterion = nn.L1Loss(reduction='none')
        else:
            criterion = nn.MSELoss(reduction='none')
        
        # Create timeslice weights: boost importance of t=10-40 where SNR is low but non-zero
        seq_len = self.model_config['max_len']
        t = torch.arange(seq_len, dtype=torch.float32, device=self.device)
        # Gaussian-like weighting centered at t=20, plus baseline
        time_weights = 1.0 + 5.0 * torch.exp(-((t - 20) ** 2) / (2 * 15 ** 2))
        time_weights = time_weights / time_weights.mean()  # normalize
        self.time_weights = time_weights.view(1, -1, 1)  # (1, seq_len, 1)
        
        #LR scheduler for faster convergence:
        if not self.training_config['scheduler']['use_scheduler']:
            scheduler = None
        else:
            if self.training_config['scheduler']['type'] == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimiser,
                    patience = self.training_config['scheduler']['patience'],
                    factor = self.training_config['scheduler']['factor'],
                    mode = self.training_config['scheduler']['mode']
                )
            elif self.training_config['scheduler']['type'] == 'StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimiser,
                    step_size = self.training_config['scheduler']['step_size'],
                    gamma = self.training_config['scheduler']['gamma']
                )

        # Gradient clipping:
        if self.training_config['gradient_clipping']['use_clipping']:
            max_grad_norm = self.training_config['gradient_clipping']['max_norm']

        # Warmup scheduler setup:
        warmup_epochs = self.training_config['scheduler'].get('warmup_epochs', 0)
        warmup_scheduler = None

        if warmup_epochs > 0:
            def warmup_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs 
                return 1.0
            
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=warmup_lambda)
            print(f"Using LR warmup for {warmup_epochs} epochs")

        #Training tracking variables:
        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None

        # TensorBoard writer - save to results directory
        writer_dir = self.results_dir / 'tensorboard_logs'
        writer_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(writer_dir))

        writer.add_text('Hyperparameters', str({
            'd_model' : self.model_config['d_model'],
            'num_layers' : self.model_config['num_layers'],
            'num_heads' : self.model_config['num_heads'],
            'dropout' : self.model_config['dropout'],
            'learning_rate' : self.training_config['learning_rate'],
            'batch_size' : self.training_config['batch_size']
        }))

        #dataloader setup for training and evaluation:
        train_loader = trainloader
        eval_loader = evalloader

        # Training loop:
        for epoch in range(int(self.training_config['num_epochs'])):
            self.train() #train mode
            total_train_loss = 0.0

            for batch_idx, batch_data in enumerate(train_loader):
                #Move data to relevant device for training:
                inputs = batch_data['features'].to(self.device)
                targets = batch_data['targets'].to(self.device) 
                #zero gradients:
                optimiser.zero_grad()
                #Forward pass:
                predictions = self.forward(inputs)
                
                # Apply timeslice weights to loss
                raw_loss = criterion(predictions, targets)
                seq_len = predictions.shape[1]
                weights = self.time_weights[:, :seq_len, :]
                value_loss = (raw_loss * weights).mean()
                
                pred_derivative = predictions[:, 1:, :] - predictions[:, :-1, :]
                target_derivative = targets[:, 1:, :] - targets[:, :-1, :]
                slope_loss = criterion(pred_derivative, target_derivative).mean()
                loss = value_loss + 2.0 * slope_loss

                #Backprop:
                loss.backward()
                #Gradient clipping if enabled:
                if self.training_config['gradient_clipping']['use_clipping']:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                #optimiser:
                optimiser.step()
                total_train_loss += loss.item()
            # Average training loss for epoch:
            avg_train_loss = total_train_loss / len(train_loader)

            #Validation:
            self.eval() #eval mode
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
            
            # Warmup phase
            if warmup_scheduler is not None and epoch < warmup_epochs:
                warmup_scheduler.step()
            # Main scheduler phase (after warmup)
            elif scheduler is not None:
                if self.training_config['scheduler']['type'] == 'ReduceLROnPlateau':
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            # Get current learning rate
            current_lr = optimiser.param_groups[0]['lr']
            
            #training history logging:
            writer.add_scalar('Loss/Train', avg_train_loss, epoch+1) #training loss tracking
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch+1) #validation loss tracking
            writer.add_scalar('learning_rate', current_lr, epoch+1) #learning rate tracking
            writer.add_scalars('Loss/Train_vs_Validation', {
                'Train' : avg_train_loss,
                'Validation' : avg_val_loss},
                epoch+1)

            #Print epoch summary:
            print(f"Epoch [{epoch+1}/{self.training_config['num_epochs']}], "
            f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
            #Early stopping and model saving:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Store best state dict in memory (save only at end)
                best_state_dict = {k: v.cpu().clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.training_config['early_stopping_patience']:
                    print("Early stopping triggered.")
                    writer.add_text('Training Stopped', f"Early stopping at epoch {epoch+1}")
                    break
        
        #Logging final metrics:
        writer.add_hparams({
            'd_model' : self.model_config['d_model'],
            'num_layers' : self.model_config['num_layers'],
            'num_heads' : self.model_config['num_heads'],
            'lr' : self.training_config['learning_rate'],
        },
        {
            'best_val_loss' : best_val_loss
        })
        writer.close() #close the tensorboard writer
        
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

    def predict_model(self, testloader, target_scaler:Optional[StandardScaler]=None, 
                      bias_vector: Optional[np.ndarray]=None, save_predictions=True):
        """Generates predictions on test data after training. Does not compute metrics for evaluation.
           Use evaluate method for computing metrics based on predictions.
           
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
        
        # Inverting standard scaling if scaler provided
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
            all_predictions = all_predictions - bias_vector  # Subtract bias
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

    def evaluate_model(self, testloader, target_scaler:Optional[StandardScaler]=None, 
                        bias_vector: Optional[np.ndarray]=None, save_predictions=True,
                        target_signs: Optional[np.ndarray]=None):
        """Evaluates testing data and computes metrics. Also saves metrics and predictions.
        
        Args:
            testloader: DataLoader for test data
            target_scaler: Fitted StandardScaler for inverse transform (required if bias_vector provided)
            bias_vector: Pre-computed bias correction vector of shape (seq_len,). If provided,
                         predictions will be bias-corrected before computing metrics.
            save_predictions: Whether to save predictions to disk
            target_signs: Signs array for inverse log transform (exp(x) * signs)
            
        Returns:
            If unscaled: (scaled_metrics, mean_relative_error, predictions, relative_errors)
            If scaled only: (scaled_metrics, predictions)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before model evaluation.")
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
        
        # Squeeze to (N_configs, seq_len) for metrics calculation
        predictions = predictions.squeeze(-1)
        targets = targets.squeeze(-1)
        
        # Scaled metric computation (before any transforms)
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
            
            # Apply inverse log transform: exp(x) * signs
            if target_signs is not None:
                predictions = np.exp(predictions) * target_signs
                targets = np.exp(targets) * target_signs
            
            # Apply bias correction if provided
            if bias_vector is not None:
                predictions = predictions - bias_vector
                bias_corrected = True
                print(f"\nApplied bias correction to predictions.")
            
            # Relative error metrics computation
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
        if not self.is_trained and not override:
            raise ValueError("Model must be trained before saving.")
        elif not self.logging_config['save_model']:
            raise ValueError("Model saving is disabled in the transformer configuration.")
        
        # Default save path to model_dir
        if save_path is None:
            save_path = self.model_dir / 'model.pth'
        else:
            save_path = Path(save_path)
        
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path=None):
        # Default load path to model_dir
        if load_path is None:
            load_path = self.model_dir / 'best_model.pth'
        else:
            load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        self.load_state_dict(torch.load(load_path, map_location=self.device))
        self.is_trained = True
        print(f"Model loaded from {load_path}")

