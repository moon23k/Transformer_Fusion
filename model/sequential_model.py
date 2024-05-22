import torch
import torch.nn as nn
from collections import namedtuple
from .components import (
    clones, load_ple, Mapping,
    MultiHeadAttention,
    PositionwiseFeedForward
)



class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(config)
        self.ple_attn = MultiHeadAttention(config)
        self.pff = PositionwiseFeedForward(config)


    def forward(self, x, ple_out, e_mask):

        s_out = self.self_attn(x, key_padding_mask=e_mask)
        p_out = self.ple_attn(x, ple_out, key_padding_mask=e_mask)
        out = x + s_out * 0.5 + p_out * 0.5

        return out + self.pff(out)




class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(config)
        self.ple_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.pff = PositionwiseFeedForward(config)


    def forward(self, x, memory, ple_out, e_mask=None, d_mask=None):
        
        m, p = memory, ple_out

        s_out = self.self_attn(x, attn_mask=d_mask)
        x = x + s_out

        p_out = self.ple_attn(x, p, key_padding_mask=e_mask)
        c_out = self.cross_attn(x, m, key_padding_mask=e_mask)
        out = x + p_out * 0.5 + c_out * 0.5

        return out + self.pff(out)



class Encoder(nn.Module):
    def __init__(self, config, ple_embeddings):
        super(Encoder, self).__init__()

        self.embeddings = ple_embeddings
        self.mapping = Mapping(config.emb_dim, config.hidden_dim)

        self.layers = clones(EncoderLayer(config), config.n_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)


    def forward(self, x, ple_out, e_mask):
        x = self.mapping(self.embeddings(x))
        for layer in self.layers:
            x = layer(x, ple_out, e_mask)
        return self.norm(x)



class Decoder(nn.Module):
    def __init__(self, config, ple_embeddings):
        super(Decoder, self).__init__()

        self.embeddings = ple_embeddings
        self.mapping = Mapping(config.emb_dim, config.hidden_dim)
        
        self.layers = clones(DecoderLayer(config), config.n_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        

    def forward(self, x, memory, ple_out, e_mask=None, d_mask=None):
        x = self.mapping(self.embeddings(x))
        for layer in self.layers:
            x = layer(x, memory, ple_out, e_mask, d_mask)
        return self.norm(x)



class SequentialModel(nn.Module):
    def __init__(self, config):
        super(SequentialModel, self).__init__()

        self.device = config.device
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size

        self.ple = load_ple(config)
        self.mapping = Mapping(self.ple.config.hidden_size, config.hidden_dim)
        
        self.encoder = Encoder(config, self.ple.embeddings)
        self.decoder = Decoder(config, self.ple.embeddings)
        self.generator = nn.Linear(config.hidden_dim, self.vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')


    @staticmethod    
    def shift_y(x):
        return x[:, :-1], x[:, 1:]    


    def mask(self, x, y=None):
        x_mask = x == self.pad_id

        if y is None:
            return x_mask

        sz = y.size(1)
        y_mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)

        return x_mask, y_mask


    def encode(self, x, ple_out, e_mask):        
        return self.encoder(x, ple_out, e_mask)
    

    def decode(self, x, m, ple_out, e_mask, d_mask=None):
        return self.decoder(x, m, ple_out, e_mask, d_mask)


    def forward(self, input_ids, attention_mask, labels):        
        x = input_ids
        y, label = self.shift_y(labels)
        e_mask, d_mask = self.mask(x, y)

        ple_out = self.ple(input_ids=x, attention_mask=attention_mask).last_hidden_state
        ple_out = self.mapping(ple_out)

        memory = self.encode(x, ple_out, e_mask)
        d_out = self.decode(y, memory, ple_out, e_mask, d_mask)
        
        logit = self.generator(d_out)
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )

        return self.out

