import torch
import torch.nn as nn

from src.models.inits_custom import reset


EPS = 1e-15

class FATEMaskedAE(torch.nn.Module):

    def __init__(self, encoder, 
                 loss_ftn,  
                 n_marker, 
                 latent_dim,
                 pos_encoding_dim,
                 encoder_out_dim, 
                 decoder,
                 pred_head = None,
                 supervision: bool = False):
        super().__init__()

        # Models
        self.encoder = encoder
        self.decoder = decoder
        self.pred_head = pred_head

        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.pos_encoding = nn.Parameter(torch.randn(n_marker, pos_encoding_dim))

        
        # Flags
        self.supervision = supervision

        # AE loss used
        self.loss_ftn = loss_ftn
    
       
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)
        if self.supervision:
            reset(self.pred_head)



    def forward(self, x, marker_idx_encoder, marker_idx_decoder, *args, **kwargs):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""

        encoded_x = self.encoder(x, self.pos_encoding[marker_idx_encoder])
        prediction = None

        z= self.fc_mu(encoded_x)


        if self.supervision:
            prediction = self.pred_head(z)


        return  z, self.decoder(self.pos_encoding[marker_idx_decoder],z), prediction
 
    def ae_loss(self, output, target):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = output
        input = target

        recons_loss = self.loss_ftn(recons, input)#mse_loss_torch(recons, input)

        return recons_loss


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'







