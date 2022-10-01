import torch
import torch.nn as nn
from models.base import get_clones, DecoderLayer
from transformers import BertModel



class Encoder(nn.Module):
    def __init__(self, config):
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.embeddings = self.bert.embeddings
        self.fc = nn.Linear(config.bert_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x):
        out = self.bert(x)['last_hidden_state']
        return self.dropout(self.fc(out))



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
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config, self.encoder.embeddings)
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)


    def forward(self, src, trg):
        out = self.decoder(self.embeddings(trg), 
                           self.encoder(src), 
                           self.pad_mask(src), 
                           self.dec_mask(trg))
        
        return self.fc_out(out)

    
    def pad_mask(self, x):
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
    
    def dec_mask(self, x):
        pad_mask = self.pad_mask(x)
        sub_mask = torch.tril(torch.ones((x.size(-1), x.size(-1)))).bool().to(self.device)
        return pad_mask & sub_mask