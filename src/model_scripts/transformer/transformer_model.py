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
        project_root = Path(__file__).parent.parent.parent #MLLQCD/
        self.data_path = project_root / self.config['data']['input_data_path'] / experiment_folder
        self.results_dir = project_root / 'results' / experiment_folder
        self.results_dir.mkdir(parents=True, exist_ok=True)
        

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
    
    def train_model(self, X_train, y_train, X_val, y_val):
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

        writer_dir = Path('runs') / self.training_config['tensorboard_log_dir']
        writer = SummaryWriter(log_dir=writer_dir)

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
        
            #Early stopping:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if self.logging_config['save_best_model']:
                    checkpoint_path = Path(self.training_config.get('checkpoint_dir', 'checkpoints'))
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    self.save_model(checkpoint_path / 'best_model.pth', override=True)
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
        self.is_trained = True
        print(f"Training completed.")
    
    def bias_correction(self, bias_corr_data):
        #Bias correction logic post training and eval step will be over here
        pass

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
        print(f"Evaluation Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name} : {metric_value:.6f}")
        return metrics, predictions

    def save_model(self, save_path, override=False):
        if not self.is_trained and not override:
            raise ValueError("Model must be trained before saving.")
        elif not self.logging_config['save_model']:
            raise ValueError("Model saving is disabled in the transformer configuration.")
        else:
            torch.save(self.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    
    def load_model(self, load_path):
        self.load_state_dict(torch.load(load_path))
    


