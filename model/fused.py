import torch
import torch.nn as nn
from transformers import BertModel
from model.simple import (clones, 
                          LayerNorm,
                          Embeddings, 
                          MultiHeadAttention,
                          PositionwiseFeedForward)


class Sublayer(nn.Module):
    def __init__(self, config):
        super(Sublayer, self).__init__()
        self.norm = LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x, sublayer):
        return self.dropout(sublayer(self.norm(x)))



class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(config)
        self.bert_attn = MultiHeadAttention(config)
        self.pff = PositionwiseFeedForward(config)

        self.s_sublayer = Sublayer(config)
        self.b_sublayer = Sublayer(config)
        self.p_sublayer = Sublayer(config)


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

        self.s_sublayer = Sublayer(config)
        self.b_sublayer = Sublayer(config)
        self.e_sublayer = Sublayer(config)
        self.p_sublayer = Sublayer(config)


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
        self.norm = LayerNorm(config.hidden_dim)
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
        self.norm = LayerNorm(config.hidden_dim)
        self.layers = clones(DecoderLayer(config), config.n_layers)
        

    def forward(self, x, memory, e_mask, d_mask, bert_out):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, memory, e_mask, d_mask, bert_out)
        return self.norm(x)



class FusedModel(nn.Module):
    def __init__(self, config):

        self.device = config.device
        self.pad_id = config.pad_id

        self.bert = BertModel.from_pretrained(config.bert)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.fc_out = nn.Linear(config.hidden_dim, config.vocab_size)


    def pad_mask(self, x):
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)


    def dec_mask(self, x):
        seq_len = x.size(-1)
        attn_shape = (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
        return self.pad_mask(x) & subsequent_mask.to(self.device)


    def forward(self, src, trg):
        e_mask, d_mask = self.pad_mask(src), self.dec_mask(trg)
        bert_out = self.bert(src)

        memory = self.encoder(src, e_mask, bert_out)
        dec_out = self.decoder(trg, memory, e_mask, d_mask, bert_out)
        return self.fc_out(dec_out)
