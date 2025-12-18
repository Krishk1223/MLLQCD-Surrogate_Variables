import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from ..base_model import BaseModel
from .transformer_components.InputEmbedding import InputEmbedding
from .transformer_components.PositionalEncoding import PositionalEncoding
from .transformer_components.EncoderLayer import EncoderLayer
from .transformer_components.RegressionHead import RegressionHead
from src.dataloader.transformer_dataloader import TransformerDataLoader
from pathlib import Path
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json


class Transformer(BaseModel):
    def __init__(self, config, experiment_folder):
        nn.Module.__init__(self) #initialise nn.Module
        super().__init__(config) #passes config to basemodel.

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
        self.results_dir = project_root / 'results' / experiment_folder
        self.model_dir = project_root / 'models' / experiment_folder / 'transformer_model'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        #Component initialisation
        self.input_embed = InputEmbedding(
            self.model_config['input_dim'],
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
            for _ in range(self.model_config['num_layers'])
        ])
        self.regressor_head = RegressionHead(
            self.model_config['d_model'],
            self.model_config['regression_head']['output_dim'])

        self.is_built = True

    def forward(self, x):
        x = self.input_embed(x)
        x = self.pos_encoder(x)

        for layer in self.encoder_layers:
            x = layer(x)

        output = self.regressor_head(x)
        return output

    def build_model(self):
        #Components built in init but method used to verify if trained
        if not self.is_built:
            raise ValueError("Model components are not built properly.")
        self.is_trained = False
        return self
    
    def train_model(self):
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
            criterion = nn.HuberLoss() #better for outliers than MSE while remaining differentiable
        elif self.training_config['loss_function'] == 'MAE':
            criterion = nn.L1Loss() #better for outliers than MSE but less stable/differentiable
        else:
            criterion = nn.MSELoss() #default
        
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

        #Training tracking variables:
        best_val_loss = float('inf')
        patience_counter = 0

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
        train_loader = TransformerDataLoader(self.data_path, self.config, split='train')
        eval_loader = TransformerDataLoader(self.data_path, self.config, split='eval')

        training_history = []
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
                loss = criterion(predictions, targets)

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
                    loss = criterion(predictions, targets)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(eval_loader)

            #learning rate scheduler step:
            current_lr = optimiser.param_groups[0]['lr']
            if scheduler is not None:
                if self.training_config['scheduler']['type'] == 'ReduceLROnPlateau':
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            
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
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
            #Early stopping and model saving:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if self.logging_config['save_best_model']:
                    # Save to model_dir instead of checkpoint_dir
                    best_model_path = self.model_dir / 'best_model.pth'
                    self.save_model(best_model_path, override=True)
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
        
        # Save final model
        final_model_path = self.model_dir / 'final_model.pth'
        self.save_model(final_model_path, override=True)
        
        self.is_trained = True
        print(f"Training completed.")
        print(f"Best model saved to: {self.model_dir / 'best_model.pth'}")
        print(f"Final model saved to: {final_model_path}")
        print(f"TensorBoard logs saved to: {writer_dir}")
    
    def compute_bias_correction(self):
        """Bias correction method to adjust model predictions."""
        bias_loader = TransformerDataLoader(self.data_path, self.config, split='bias')
        self.eval() #eval mode
        with torch.no_grad():
            y_pred = 


    def predict_model(self):
        """Generates predictions on test data after training. Does not compute metrics for evaluation.
           Use evaluate method for computing metrics based on predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        testloader = TransformerDataLoader(self.data_path, self.config, split='test')
        all_predictions = []
        self.eval() #eval mode
        with torch.no_grad():
            for batch_data in testloader:
                inputs = batch_data['features'].to(self.device)
                predictions = self.forward(inputs)
                all_predictions.append(predictions.cpu()) #move to cpu for aggregation

        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        
        # Save predictions to results directory
        predictions_path = self.results_dir / 'test_predictions.npy'
        np.save(predictions_path, all_predictions)
        print(f"Test predictions saved to: {predictions_path}")
        
        return all_predictions #returns tensor of predictions

    def evaluate_model(self):
        """Evaluates testing data after predict method and computes metrics."""
        if not self.is_trained:
            raise ValueError("Model must be trained before model evaluation.")
        predictions = []
        targets = []
        self.eval() #eval mode
        testloader = TransformerDataLoader(self.data_path, self.config, split='test')
        with torch.no_grad():
            for batch_data in testloader:
                inputs = batch_data['features'].to(self.device)
                target = batch_data['targets'].to(self.device)
                output = self.forward(inputs)
                predictions.append(output.cpu())
                targets.append(target.cpu())
        predictions = torch.cat(predictions, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

        #metric computation:
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        metrics = {
            'Mean Squared Error' : mse,
            'Root Mean Squared Error' : rmse,
            'Mean Absolute Error' : mae,
            'R2 Score' : r2
        }
        
        print(f"\nEvaluation Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name} : {metric_value:.6f}")
        
        # Save metrics to results directory
        metrics_path = self.results_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Evaluation metrics saved to: {metrics_path}")
        
        # Save predictions
        np.save(self.results_dir / 'test_predictions.npy', predictions)
        print(f"Test predictions saved to: {self.results_dir / 'test_predictions.npy'}")
        return metrics, predictions

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



