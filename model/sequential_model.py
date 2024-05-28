import torch
import torch.nn as nn
from .components import (
    clones, LayerBase, ModelBase,
    SublayerConnection,
    PositionwiseFeedForward, 
)





class EncoderLayer(LayerBase):
    def __init__(self, config):
        super(EncoderLayer, self).__init__(config)

        #Common Setups
        self.self_attn = nn.MultiheadAttention(**self.attn_params)
        self.pff = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)


        #Setups for Parallel Fusion Process
        if self.enc_fuse:
            self.ple_mapping = nn.Sequential(
                nn.Linear(config.ple_hidden_dim, config.hidden_dim),
                nn.Dropout(config.dropout_ratio)
            )
            self.ple_attn = nn.MultiheadAttention(**self.attn_params)


    def forward(self, x, ple_out, e_mask):

        if self.enc_fuse:
            s_out = self.self_attn(x, key_padding_mask=e_mask)
            p_out = self.ple_attn(x, ple_out, key_padding_mask=e_mask)
            out = x + s_out * 0.5 + p_out * 0.5
            return out + self.pff(out)
        
        else:
            out = self.self_attn(x, key_padding_mask=e_mask)
            return out + self.pff(out)




class DecoderLayer(LayerBase):
    def __init__(self, config):
        super(DecoderLayer, self).__init__(config)

        #Common Setups
        self.self_attn = nn.MultiheadAttention(**self.attn_params)
        self.cross_attn = nn.MultiheadAttention(**self.attn_params)
        self.pff = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 3)


        #Setups for Parallel Fusion Process
        if self.dec_fuse:
            self.ple_mapping = nn.Sequential(
                nn.Linear(config.ple_hidden_dim, config.hidden_dim),
                nn.Dropout(config.dropout_ratio)
            )
            self.ple_attn = nn.MultiheadAttention(**self.attn_params)




    def forward(self, x, memory, ple_out, e_mask=None, d_mask=None):
        
        if self.dec_fuse:
            m, p = memory, ple_out

            s_out = self.self_attn(x, attn_mask=d_mask)
            x = x + s_out

            p_out = self.ple_attn(x, p, key_padding_mask=e_mask)
            c_out = self.cross_attn(x, m, key_padding_mask=e_mask)
            out = x + p_out * 0.5 + c_out * 0.5

            return out + self.pff(out)
        
        else:
            return



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.emb_mapping = nn.Sequential(
            nn.Linear(config.emb_dim, config.hidden_dim),
            nn.Dropout(config.dropout_ratio)
        )
        self.layers = clones(EncoderLayer(config), config.n_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)



    def forward(self, x, ple_out=None, e_mask=None):

        x = self.emb_mapping(x)

        for layer in self.layers:
            x = layer(x, ple_out, e_mask)

        return self.norm(x)




class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.emb_mapping = nn.Sequential(
            nn.Linear(config.emb_dim, config.hidden_dim),
            nn.Dropout(config.dropout_ratio)
        )
        
        self.layers = clones(DecoderLayer(config), config.n_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        


    def forward(self, x, memory, ple_out=None, e_mask=None, d_mask=None):
        
        x = self.emb_mapping(x)
        
        for layer in self.layers:
            x = layer(x, memory, ple_out, e_mask, d_mask)

        return self.norm(x)




class SequentialModel(ModelBase):
    def __init__(self, config, ple):
        super(SequentialModel, self).__init__(config, ple)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.generator = nn.Linear(config.hidden_dim, self.vocab_size)



    def forward(self, input_ids, attention_mask, labels):
        #Prerequisites
        x = input_ids
        y, label = self.shift_y(labels)
        e_mask, d_mask = self.pad_mask(x), self.causal_mask(y)

        #Embedding
        x = self.ple.embeddings(x)
        y = self.ple.embeddings(y)

        #Actual Process
        ple_out = self.ple(input_ids=x, attention_mask=attention_mask).last_hidden_state
        memory = self.encoder(x, ple_out if self.enc_fuse else None, e_mask)
        d_out = self.decoder(y, memory, ple_out if self.dec_fuse else None, e_mask, d_mask)
        logit = self.generator(d_out)
        
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )

        return self.out

