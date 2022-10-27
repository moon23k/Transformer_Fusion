import torch
import torch.nn as nn
import transformers
from models.transformer import Transformer



class FusedModel(nn.Module):
	def __init__(self, config):
		super(FusedModel, self).__init__()
		self.bert = config.bert
		self.embeddings = self.bert.embeddings
		self.encoder = Transformer.Encoder(config)
		self.decoder = Transformer.Decoder(config)


	def forward(self, src, trg):
		
		return




class AddTopModel(nn.Module):
	def __init__(self, config):
		super(AddTopModel, self).__init__()
		self.bert = config.bert
		self.fc_out = nn.Linear()


	def forward(self, src, trg):
		return self.fc_out(self.bert(src, trg))