import torch
import torch.nn as nn
from transformers import BertModel
from models.transformer import clones, Embeddings, EncoderLayer, DecoderLayer, Decoder



class FusedEncoder(nn.Module):
	def __init__(self, config):
		super(FusedEncoder, self).__init__()
		self.embeddings = Embeddings(config)
		self.layers = clones()


class FusedDecoder(nn.Module):
	def __init__(self, config):
		super(FusedDecoder, self).__init__()
		self.embeddings = Embeddings(config)
		self.layers = clones()



class FusedModel(nn.Module):
	def __init__(self, config):
		super(FusedModel, self).__init__()
		self.bert = config.bert
		self.embeddings = self.bert.embeddings
		self.encoder = FusedEncoder(config)
		self.decoder = FusedDecoder(config)


    def pad_mask(self, x):
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def dec_mask(self, x):
        seq_len = x.size(-1)
        attn_shape = (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
        return self.pad_mask(x) & subsequent_mask.to(self.device)

	def forward(self, src, trg):
		
		return




class SimpleModel(nn.Module):
    def __init__(self, config):
        super(SimpleModel, self).__init__()
        self.encoder = BertModel.from_pretrained(config.bert_model)
        self.decoder = Decoder(config)
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)

    def pad_mask(self, x):
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def dec_mask(self, x):
        seq_len = x.size(-1)
        attn_shape = (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
        return self.pad_mask(x) & subsequent_mask.to(self.device)

    def forward(self, src, trg):
        e_mask, d_mask = self.pad_mask(src), self._dec_mask(trg)
        memory = self.encoder(src, e_mask)[0]
        out = self.decoder(trg, memory, e_mask, d_mask)
        return self.fc_out(out)