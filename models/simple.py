import torch
import torch.nn as nn
from models.base import get_clones, DecoderLayer
from transformers import BertModel


class Decoder(nn.Module):
    def __init__(self, config, embeddings):
        super(Decoder, self).__init__()
        self.embeddings = embeddings
        self.layers = get_clones(DecoderLayer(config), config.n_layers)
    
    def forward(self, trg, memory, src_mask, trg_mask):
        trg = self.embeddings(trg)
        for layer in self.layers:
            trg = layer(trg, memory, src_mask, trg_mask)
        return trg


class SimpleModel(nn.Module):
    def __init__(self, config):
        super(SimpleModel, self).__init__()
        self.device = config.device
        self.pad_idx = config.pad_idx
        
        self.bert = BertModel.from_pretrained('bert-base-cased')
        #self.bert.resize_token_embeddings(config.input_dim)
        self.embeddings = self.bert.embeddings

        self.decoder = Decoder(config, self.embeddings)
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, src, trg):
        src_mask, trg_mask = self.pad_mask(src), self.dec_mask(trg)
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(trg, enc_out, src_mask, trg_mask)
        return self.fc_out(dec_out)
    
    def pad_mask(self, x):
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
    
    def dec_mask(self, x):
        pad_mask = self.pad_mask(x)
        sub_mask = torch.tril(torch.ones((x.size(-1), x.size(-1)))).bool().to(self.device)
        return pad_mask & sub_mask