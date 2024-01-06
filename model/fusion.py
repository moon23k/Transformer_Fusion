import torch
import torch.nn as nn
from collections import namedtuple
from transformers import AutoModel
from .common import (
    clones, 
    Embeddings, 
    MultiHeadAttention,
    PositionwiseFeedForward,
    SublayerConnection
)



class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(config)
        self.bert_attn = MultiHeadAttention(config)
        self.pff = PositionwiseFeedForward(config)

        self.s_sublayer = SublayerConnection(config) #self attn
        self.b_sublayer = SublayerConnection(config) #bert attn
        self.p_sublayer = SublayerConnection(config) #pff


    def forward(self, x, mask, bert_out):
        b = bert_out

        #BERT Attn & Self Attn
        residual = x
        b = self.b_sublayer(x, lambda x: self.bert_attn(x, b, b, mask))
        s = self.s_sublayer(x, lambda x: self.self_attn(x, x, x, mask))
        x = residual + s * 0.5 + b * 0.5  #residual conn

        #Position wise FFN
        residual = x
        x = self.p_sublayer(x, self.pff)
        return residual + x  #residual conn




class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(config)
        self.bert_attn = MultiHeadAttention(config)
        self.enc_dec_attn = MultiHeadAttention(config)
        self.pff = PositionwiseFeedForward(config)

        self.s_sublayer = SublayerConnection(config) #self
        self.b_sublayer = SublayerConnection(config) #bert
        self.e_sublayer = SublayerConnection(config) #encoder
        self.p_sublayer = SublayerConnection(config) #pff


    def forward(self, x, memory, e_mask, d_mask, bert_out):
        m = memory
        b = bert_out

        #Self Attn
        residual = x
        s = self.s_sublayer(x, lambda x: self.self_attn(x, x, x, d_mask))
        x = residual + s  #residual conn
        

        #BERT Attn & Enc-Dec Attn
        residual = x
        b = self.b_sublayer(x, lambda x: self.bert_attn(x, b, b, e_mask))
        e = self.b_sublayer(x, lambda x: self.bert_attn(x, m, m, e_mask))
        x = residual + b * 0.5 + e * 0.5  #residual conn
        

        #Position wise FFN
        residual = x
        x = self.p_sublayer(x, self.pff)
        return residual + x  #residual conn



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.emb = Embeddings(config)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.layers = clones(EncoderLayer(config), config.n_layers)

    def forward(self, x, mask, bert_out):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, mask, bert_out)
        return self.norm(x)



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.emb = Embeddings(config)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.layers = clones(DecoderLayer(config), config.n_layers)
        

    def forward(self, x, memory, e_mask, d_mask, bert_out):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, memory, e_mask, d_mask, bert_out)
        return self.norm(x)



class FusionModel(nn.Module):
    def __init__(self, config):
        super(FusionModel, self).__init__()
        
        self.device = config.device
        self.pad_id = config.pad_id
        self.max_len = config.max_len
        self.vocab_size = config.vocab_size

        self.ple = AutoModel.from_pretrained(config.mname)
        self.encoder = Encoder(config)
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
        ple_out = self.ple(input_ids, attention_mask).last_hidden_state
        memory = self.encoder(input_ids, e_mask, ple_out)

        preds = torch.zeros(batch_size, self.max_len).to(self.device)
        for i in range(1, self.max_len):
            d_mask = self.dec_mask(preds)
            dec_out = self.decoder(preds, memory, e_mask, d_mask)
            logits = self.generator(dec_out).argmax(-1)

            if logits.sum() == 0:
                break

            preds[i] = logits

        return preds.tolist()


    def forward(self, input_ids, attention_mask, labels):
        y = self.shift_right(labels)

        e_mask = self.pad_mask(input_ids)
        d_mask = self.dec_mask(y)
        
        ple_out = self.ple(input_ids, attention_mask).last_hidden_state

        memory = self.encoder(input_ids, e_mask, ple_out)
        d_out = self.decoder(y, memory, e_mask, d_mask, ple_out)
        
        logits = self.generator(d_out)
        loss = self.criterion(logits.view(-1, self.vocab_size), 
                              labels[:, 1:].contiguous().view(-1))

        return self.outputs(logits, loss)
