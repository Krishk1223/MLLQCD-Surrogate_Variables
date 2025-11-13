import torch
import torch.nn as nn
from ..base_model import BaseModel
from .transformer_components.InputEmbedding import InputEmbedding
from .transformer_components.PositionalEncoding import PositionalEncoding
from .transformer_components.EncoderLayer import EncoderLayer
from .transformer_components.RegressionHead import RegressionHead
import yaml


class Transformer(BaseModel):
    def __init__(self, config):
        super().__init__(config) #passes config to basemodel
        nn.Module.__init__(self)

        #config:
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        self.logging_config = config['logging'] #will set up logging later

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
    
    def train(self, train_data, val_data):
        #Training logic will be implemented here
        super().train() #model in training mode

        #Device setup:
        if torch.cuda.is_available():
            device = torch.device('cuda') #use CUDA if available
        elif torch.backends.mps.is_available():
            device = torch.device('mps') # use metal performance shaders for macs with apple silicon
        else:
            device = torch.device('cpu') #cpu if nothing else available
        self.to(device)

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
    
        #Training tracking variables:
        best_val_loss = float('inf')
        patience_counter = 0
        

        self.is_trained = True
    
    def bias_correction(self, bias_corr_data):
        #Bias correction logic post training and eval step will be over here
        pass

    def predict(self, test_data):
        #Prediction logic will be implemented here
        pass
        # self.eval()
        # with torch.no_grad():
        #     return self.forward(test_data)

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)
    
    def load_model(self, load_path):
        self.load_state_dict(torch.load(load_path))
    


