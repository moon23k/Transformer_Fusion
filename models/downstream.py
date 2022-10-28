import torch
import torch.nn as nn
from transformers import BertModel
from models.transformer import clones, Embeddings, EncoderLayer, DecoderLayer



class Encoder(nn.Module):
	def __init__(self, config):
		super.(Encoder, self).__init__()
		self.embeddings = Embeddings(config)
		self.layers = get_clone


class Deocder(nn.Module)
	def __init__(self, config):
		super.(Encoder, self).__init__()
		self.embeddings = Embeddings(config)
		self.layers = get_clone



class FusedModel(nn.Module):
	def __init__(self, config):
		super(FusedModel, self).__init__()
		self.bert = config.bert
		self.embeddings = self.bert.embeddings
		self.encoder = Encoder(config)
		self.decoder = Decoder(config)


	def pad_mask(self, x):
		return


	def dec_mask(self, x):
		return


	def forward(self, src, trg):
		
		return




class AddTopModel(nn.Module):
	def __init__(self, config):
		super(AddTopModel, self).__init__()
		self.bert = BertModel.from_pretrained(config.bert_model)
		self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)

	def forward(self, src, trg):
		out = self.bert(src, trg)[0]
		return self.fc_out(out)