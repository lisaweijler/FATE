import torch
import numpy as np

from src.models.inits_custom import reset
from src.models.base_model import BaseModel

class SupervisedModel(BaseModel):

    def __init__(self, 
                 encoder, 
                 pred_head, 
                 n_marker, 
                 pos_encoding_dim,
                 encoder_out_dim, 
                 latent_dim):
        super().__init__()
        self.encoder = encoder

        self.pos_encoding = torch.nn.Parameter(torch.randn(n_marker, pos_encoding_dim))
        self.fc_mu = torch.nn.Linear(encoder_out_dim, latent_dim)

        self.pred_head = pred_head       
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.pred_head)
    
    def forward(self, x, marker_idx_encoder):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""

        encoded_x = self.encoder(x, self.pos_encoding[marker_idx_encoder])

        z = self.fc_mu(encoded_x)
        
        return  self.pred_head(z), z



   







