import torch.nn as nn
import torch

from src.models.modules import ISAB, MAB, PMA
from src.models.base_model import BaseModel


class EventEmbedderType(nn.Module):
    """
    The module that embeds the Event in a general Space - using one seed vector of target dimension.

    """
    def __init__(self, 
                 pos_encoding_dim: int, 
                 dim_event_embedding: int,
                 num_heads_embedder: int, 
                 layer_norm: bool = False):
        super().__init__()

        self.eventwise_marker_attention = MAB(1+pos_encoding_dim, 1+pos_encoding_dim, dim_event_embedding, num_heads=num_heads_embedder, ln=layer_norm) # self attention # 2 is dim input since we have marker value and pos encoding, now it is marker value plus
        self.eventwise_embedder = PMA(dim_event_embedding, num_heads=num_heads_embedder, num_seeds=1, ln=layer_norm)
       

    def forward(self, x, pos_encodings):
        # x has dim n_events x n_marker
        # pos_encodings is matrix of shape n_events x n_marker x pos_embedding_dim

        # split in single events -> batch dim(= n_events) x n_marker x 1; transpose faster than cat
        x_eventwise = x.unsqueeze(0).transpose(1,2).transpose(0,2) 

        # add pos encoding of marker to x_eventwise -> n_events (= e.g. 600) is batchsize, n_marker = 14 and pos_enc_dim+1 = 11 -> [600, 14,11]
        x_eventwise_pos_m = torch.cat((x_eventwise, pos_encodings), dim=2)

        # Eventwise self-attention between marker -> n_events x n_marker x embedding_dim: [600, 14,2]  if embedding dim = 2
        x_embedded_ = self.eventwise_marker_attention(x_eventwise_pos_m, x_eventwise_pos_m) 

        # Cross attention with seed vector -> n_events x 1=num_seeds x hidden_dim [600, 1, 2]
        x_embedded_ = self.eventwise_embedder(x_embedded_) 

        # reshape
        x_embedded = x_embedded_.transpose(0,1) # 1 * n_events *n_marker
        return x_embedded


class FATE(BaseModel):
    """
    Set transformer as described in https://arxiv.org/abs/1810.00825
    dim_input:  dimensionality of input              (flowdata: number of markers)
    num_ouputs: output sequence length               (flowdata: sequence length)
    num_inds:   number of induced points
    dim_hidden: dimension of hidden representation
    num_heads:  number of attention heads
    ln:         use layer norm true/false
    """
    def __init__(self,
                 dim_event_embedding,
                 num_heads_embedder, 
                 dim_hidden,        # dim_hidden must be divisible by num_heads i.e. dim_hidden%num_heads = 0
                 num_heads,
                 num_inds,
                 layer_norm,
                 pos_encoding_dim: int = 10# runter drehen für memory einsparnisse,
                 ): 
        super().__init__()

        # event embedder
        self.event_embedder = EventEmbedderType(pos_encoding_dim, dim_event_embedding, num_heads_embedder, layer_norm=True)

        # normal ST 
        self.isab1 = ISAB(dim_event_embedding, dim_hidden, num_heads, num_inds, ln=layer_norm)
        self.isab2 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=layer_norm)
        self.isab3 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=False)


    def forward(self, x, pos_encodings):
        if len(x.shape) ==3:
            x.squeeze(0)
        x_embedded = self.event_embedder(x, pos_encodings) # 1 * n_events*n_marker
        o1 = self.isab1(x_embedded)
        o2 = self.isab2(o1)
        o3 = self.isab3(o2) # 1* n_events * dim_output
        output = o3.squeeze(0) # n_events * dim_hidden

        return output


class FATEMaskedAEDecoder(BaseModel):
    """
    Set transformer as described in https://arxiv.org/abs/1810.00825
    dim_input:  dimensionality of input              (flowdata: number of markers)
    num_ouputs: output sequence length               (flowdata: sequence length)
    num_inds:   number of induced points
    dim_hidden: dimension of hidden representation
    num_heads:  number of attention heads
    ln:         use layer norm true/false
    """
    def __init__(self,
                 dim_hidden, # dim_hidden must be divisible by num_heads i.e. dim_hidden%num_heads = 0
                 num_heads,
                 num_inds,
                 dim_latent,
                 layer_norm,
                 pos_encoding_dim: int = 10 # runter drehen für memory einsparnisse
                 ): 
        super().__init__()

        
   
        # for attention across events - either relu former without lin embedding befor or channelwise attention
        # latents attention
        self.latent_attention_1 = ISAB(dim_latent, dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=layer_norm)
        self.latent_attention_2 = ISAB(dim_hidden, dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=layer_norm)
        self.latent_attention_3 = ISAB(dim_hidden, dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=layer_norm)

        # with the current implementation of multiple heads, more than one head does not make sense!! 
        # since query is split up.. well we have alinear layer befor.. so actuall coudl make sense after all.. 
        self.cross_attention = MAB(pos_encoding_dim, 1, dim_hidden, num_heads=num_heads, ln=False) #ln=layer_norm) # dim Q, dim K, dim V - q= queries (marker queries)
        self.eventwise_marker_attention = MAB(dim_hidden, dim_hidden, dim_hidden, num_heads=1, ln=False)
        self.out_ly = nn.Linear(dim_hidden, 1)


    def forward(self,  pos_encodings, latents):
        n_events = pos_encodings.shape[0]

        lats1 = self.latent_attention_1(latents.unsqueeze(0))
        lats2 = self.latent_attention_2(lats1)
        lats3 = self.latent_attention_3(lats2)

        latents_eventwise = lats3.transpose(1,2).transpose(0,2)

        #get the marker corresponding queries to reconstruct
        #this is a memory bottle neck - since batchsize = n_events,we could split it up here
        if n_events > 50000:
            x = self.cross_attention(pos_encodings[:50000,:,:], latents_eventwise[:50000,:,:])
            n_chunks = int(n_events/50000) # int does floor
            for i in range(1, n_chunks):
                x = torch.cat((x, self.cross_attention(pos_encodings[50000*i:50000*(i+1),:,:], latents_eventwise[50000*i:50000*(i+1),:,:])), dim = 0)
            x = torch.cat((x, self.cross_attention(pos_encodings[50000*n_chunks:,:,:], latents_eventwise[50000*n_chunks:,:,:])), dim = 0)
        else:
            x = self.cross_attention(pos_encodings, latents_eventwise) # n_events, n_marker, dim_hidden

        # if still too little - can split this up as well
        x_2 = self.eventwise_marker_attention(x,x) # n_events, n_marker, dim_hidden
        out = self.out_ly(x_2).squeeze(-1) # n_events, n_marker, 1
        return out


