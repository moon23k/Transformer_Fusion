import torch
import torch.nn as nn
from .components import clones, Embeddings



class EncModel(nn.Module):
	def __init__(self, config):
		self.encoder = Encoder


	def forward(self, x, y):
		return out