import copy, math, torch
import torch.nn as nn
from collections import namedtuple
from .components import clones, load_ple, Mapping




class Decoder(nn.Module):
    def __init__(self, config, ple_embeddings):
        super(Decoder, self).__init__()

        self.embeddings = ple_embeddings
        self.mapping = Mapping(config.emb_dim, config.hidden_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, memory, e_mask=None, d_mask=None):   
        x = self.mapping(self.embeddings(x))
        for layer in self.layers:
            x = layer(
                x, memory, 
                memory_key_padding_mask=e_mask,
                tgt_mask=d_mask,
            )

        return x



class SimpleModel(nn.Module):
    def __init__(self, config):
        super(SimpleModel, self).__init__()

        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size

        self.ple = load_ple(config)
        self.mapping = Mapping(self.ple.config.hidden_size, config.hidden_dim)

        self.decoder = Decoder(config, self.ple.embeddings)
        self.generator = nn.Linear(config.hidden_dim, self.vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')


    @staticmethod    
    def shift_y(x):
        return x[:, :-1], x[:, 1:]    


    def mask(self, x, y):
        x_mask = x == self.pad_id

        sz = y.size(1)
        y_mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)

        return x_mask, y_mask


    def encode(self, x, ple_mask):
        x = self.ple(input_ids=x, attention_mask=ple_mask).last_hidden_state
        return self.mapping(x)
        

    def decode(self, x, memory, e_mask, d_mask):
        return self.decoder(x, memory, e_mask, d_mask)


    def forward(self, input_ids, attention_mask, labels):
        x = input_ids
        y, label = self.shift_y(labels)
        e_mask, d_mask = self.mask(x, y)

        memory = self.encode(x, attention_mask)
        dec_out = self.decode(y, memory, e_mask, d_mask)
        logit = self.generator(dec_out)

        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )

        return self.out


