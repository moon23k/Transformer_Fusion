import copy, math, torch
import torch.nn as nn
from .components import clones




class Encoder(nn.Module):
    def __init__(self, config, ple):
        super(Encoder, self).__init__()

        self.ple = ple
        self.enc_mapping = nn.Sequential(
            nn.Linear(ple.config.hidden_size, config.hidden_dim),
            nn.Dropout(config.dropout_ratio)
        )

    def forward(self, x, e_mask=None):
        x = self.ple(input_ids=x, attention_mask=e_mask).last_hidden_state
        return self.enc_mapping(x)




class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.emb_mapping = nn.Sequential(
            nn.Linear(config.emb_dim, config.hidden_dim),
            nn.Dropout(config.dropout_ratio)
        )

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
        x = self.emb_mapping(x)

        for layer in self.layers:
            x = layer(
                x, memory, 
                memory_key_padding_mask=e_mask,
                tgt_mask=d_mask,
            )

        return x



class SimpleModel(ModelBase):
    def __init__(self, config, ple):
        super(SimpleModel, self).__init__(config)

        self.encoder = Encoder(config, ple)
        self.decoder = Decoder(config)
        self.generator = nn.Linear(config.hidden_dim, self.vocab_size)



    def forward(self, input_ids, attention_mask, labels):
        y, label = self.shift_y(labels)

        e_mask = self.pad_mask(input_ids)
        causal_mask = self.causal_mask(y)
        
        y = self.encoder.ple.embeddings(y)

        memory = self.encoder(input_ids, attention_mask)
        dec_out = self.decoder(y, memory, e_mask, causal_mask)
        logit = self.generator(dec_out)

        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )

        return self.out
