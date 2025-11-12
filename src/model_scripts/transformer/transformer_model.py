import torch
import torch.nn as nn
from ..base_model import BaseModel
from .transformer_components.InputEmbedding import InputEmbedding
from .transformer_components.PositionalEncoding import PositionalEncoding
from .transformer_components.EncoderLayer import EncoderLayer
from .transformer_components.RegressionHead import RegressionHead


class Transformer(BaseModel):
    def __init__(self, config):
        super().__init__(config) #passes config to basemodel
        nn.Module.__init__(self)

        #config:
        self.config = config

        #Component initialisation
        self.input_embed = InputEmbedding(config.input_dim, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_len, learnable=config.get('learnable_pos_encoding', False))
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.num_heads, config.d_ff, config.dropout) 
            for _ in range(config.num_layers)
        ])
        self.regressor_head = RegressionHead(config.d_model, config.output_dim)
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
        super().train()
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
    


