import torch.nn as nn

from src.models.base_model import BaseModel

class MLP(BaseModel):
    """
    simple MLP - used as prediction head for weakly supervised DGI
    dim_input:  dimensionality of input              (flowdata: number of markers)
    num_ouputs: output sequence length               (flowdata: sequence length)
    dim_hidden: dimension of hidden representation
    """
    def __init__(self, dim_input, 
                 dim_hidden,        
                 hidden_layers,
                 dim_output,
                 skip_con: bool = False):
        super().__init__()
        
        mlp_layers = [nn.Linear(dim_input, dim_hidden),
                     nn.GELU()]
        for _ in range(0, hidden_layers):
            mlp_layers.extend([nn.Linear(dim_hidden, dim_hidden),
                              nn.GELU()])
        mlp_layers.append(nn.Linear(dim_hidden, dim_output)) 
        self.mlp= nn.Sequential(*mlp_layers)

        self.skip_con = skip_con


    def forward(self, x):
        output  = self.mlp(x)
        if self.skip_con:
            output += x

            return output
        return output
 


class LinLayer(BaseModel):
    """
    simple MLP - used as prediction head for weakly supervised DGI
    dim_input:  dimensionality of input              (flowdata: number of markers)
    num_ouputs: output sequence length               (flowdata: sequence length)
    dim_hidden: dimension of hidden representation
    """
    def __init__(self, dim_input, 
                 dim_output,
                 use_bias: bool = True):
        super().__init__()
        

        self.mlp= nn.Linear(dim_input, dim_output, bias=use_bias)


    def forward(self, x):
        output  = self.mlp(x)

        return output
 
