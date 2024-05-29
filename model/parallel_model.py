import torch
import torch.nn as nn
from .components import (
    clones, LayerBase, ModelBase,
    SublayerConnection, PositionwiseFeedForward 
)





class EncoderLayer(LayerBase):
    def __init__(self, config):
        super(EncoderLayer, self).__init__(config)

        self.self_attn = nn.MultiheadAttention(**self.attn_params)
        self.pff = PositionwiseFeedForward(config)
        
        if self.enc_fuse:
            self.norm = nn.LayerNorm(config.hidden_dim)
            self.dropout = nn.Dropout(config.dropout_ratio)
            self.sublayer = SublayerConnection(config)
        else:
            self.sublayer = clones(SublayerConnection(config), 2)


    def forward(self, x, p_proj, e_mask):

        if self.enc_fuse:
            norm_x = self.norm(x)
            
            s_out = self.self_attn(
                norm_x, norm_x, norm_x, 
                key_padding_mask=e_mask, 
                need_weights=False
            )[0]

            p_out = self.ple_attn(
                norm_x, p_proj, p_proj, 
                key_padding_mask=e_mask, 
                need_weights=False
            )[0]

            x = x + self.dropout(s_out * 0.5 + p_out * 0.5)
            return self.sublayer(x, self.pff)
        
        else:
            x = self.sublayer[0](
                x, 
                lambda x: self.self_attn(
                    x, x, x, 
                    key_padding_mask=e_mask, 
                    need_weights=False
                )[0]
            )
        
            return self.sublayer[1](x, self.pff)




class DecoderLayer(LayerBase):
    def __init__(self, config):
        super(DecoderLayer, self).__init__(config)

        self.self_attn = nn.MultiheadAttention(**self.attn_params)
        self.cross_attn = nn.MultiheadAttention(**self.attn_params)
        self.pff = PositionwiseFeedForward(config)

        if self.dec_fuse:
            self.norm = nn.LayerNorm(config.hidden_dim)
            self.dropout = nn.Dropout(config.dropout_ratio)
            self.sublayer = clones(SublayerConnection(config), 2)
        else:
            self.sublayer = clones(SublayerConnection(config), 3)


    def forward(self, x, memory, p_proj, e_mask=None, d_mask=None):
        
        x = self.sublayer[0](
            x, 
            lambda x: self.self_attn(
                x, x, x, 
                attn_mask=d_mask,
                need_weights=False
            )[0]
        )

        if self.dec_fuse:
            norm_x = self.norm(x)

            p_out = self.ple_attn(
                norm_x, p_proj, p_proj, 
                key_padding_mask=e_mask, 
                need_weights=False
            )[0]

            c_out = self.cross_attn(
                norm_x, memory, memory, 
                key_padding_mask=e_mask, 
                need_weights=False
            )[0]

            x = x + self.dropout(p_out * 0.5 + c_out * 0.5)
            return self.sublayer[1](x, lambda x: self.pff(x))

        else:
            x = self.sublayer[1](
                x, 
                lambda x: self.cross_attn(
                    x, memory, memory, 
                    key_padding_mask=e_mask, 
                    need_weights=False
                )[0]
            )
            return self.sublayer[2](x, lambda x: self.pff(x))            




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




class ParallelModel(ModelBase):
    def __init__(self, config, ple):
        super(ParallelModel, self).__init__(config, ple)

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
        p_proj = self.ple_project(input_ids, attention_mask)
        memory = self.encoder(x, p_proj if self.enc_fuse else None, e_mask)
        d_out = self.decoder(y, memory, p_proj if self.dec_fuse else None, e_mask, d_mask)
        logit = self.generator(d_out)
        
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )

        return self.out

