import copy, math, torch
import torch.nn as nn
from transformers import AutoModel
from collections import namedtuple
from .common import clones, Embeddings





class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.pff_dim,
            dropout=config.dropout_ratio,
            activation='gelu',
            batch_first=True
        )

        self.embeddings = Embeddings(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, memory, e_mask=None, d_mask=None):
        
        x = self.embeddings(x)
        
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
        self.max_len = config.max_len
        self.vocab_size = config.vocab_size

        self.encoder = BertModel.from_pretrained(config.plm_mname)
        self.decoder = Decoder(config)
        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.outputs = namedtuple('outputs', ('logits', 'loss'))


    def pad_mask(self, x):
        return (x != self.pad_id).unsqueeze(1).unsqueeze(2)


    def dec_mask(self, x):
        seq_len = x.size(-1)
        attn_shape = (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
        return self.pad_mask(x) & subsequent_mask.to(self.device)


    def shift_right(self, labels):
        shifted = labels.new_zeros(labels.size(0), labels.size(1)-1)
        shifted = labels[:, :-1].clone()
        #shifted[:, 0] = self.pad_id #or self.decoder_start_token_id
        return shifted


    def generate(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        
        e_mask = self.pad_mask(input_ids)
        
        memory = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state

        preds = torch.zeros(batch_size, self.max_len, dtype=torch.long)
        preds = preds.fill_(self.pad_id).to(self.device)
        
        for i in range(1, self.max_len):
            d_mask = self.dec_mask(preds)
            dec_out = self.decoder(preds, memory, e_mask, d_mask)
            logits = self.generator(dec_out).argmax(-1)
            
            if logits.sum() == 0:
                break

            preds[i] = logits

        return preds.tolist()


    def forward(self, input_ids, attention_mask, labels):
        shifted_labels = self.shift_right(labels)
        e_mask = self.pad_mask(input_ids)
        d_mask = self.dec_mask(shifted_labels)
        
        memory = self.encoder(input_ids=input_ids, 
                              attention_mask=attention_mask).last_hidden_state
        d_out = self.decoder(shifted_labels, memory, e_mask, d_mask)
        
        logits = self.generator(d_out)
        loss = self.criterion(logits.view(-1, self.vocab_size), 
                              labels[:, 1:].contiguous().view(-1))

        return self.outputs(logits, loss)
