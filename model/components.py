import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple





def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        
        self.w_1 = nn.Linear(config.hidden_dim, config.pff_dim)
        self.w_2 = nn.Linear(config.pff_dim, config.hidden_dim)
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout1 = nn.Dropout(config.dropout_ratio)
        self.dropout2 = nn.Dropout(config.dropout_ratio)


    def forward(self, x):
        x = self.norm(x)
        x = self.w_2(self.dropout1(F.gelu(self.w_1(x))))
        return self.dropout2(x)




class SublayerConnection(nn.Module):
    def __init__(self, config):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))




class LayerBase(nn.Module):
    def __init__(self, config):
        super(LayerBase, self).__init__()
        
        self.enc_fuse = 'enc' in config.fusion_part
        self.dec_fuse = 'dec' in config.fusion_part

        self.attn_params = {
            'embed_dim': config.hidden_dim,
            'num_heads': config.n_heads,
            'batch_first': True
        }




class ModelBase(nn.Module):
    def __init__(self, config):
        super(ModelBase, self).__init__()

        #Attr Setup
        self.device = config.device
        self.pad_id = config.pad_id
        self.vocab_size = config.vocab_size
        self.fusion_part = config.fusion_part
        self.enc_fuse = 'enc' in config.fusion_part
        self.dec_fuse = 'dec' in config.fusion_part


        #Output Setup
        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')


    @staticmethod    
    def shift_y(x):
        return x[:, :-1], x[:, 1:]


    def pad_mask(self, x):
        return x == self.pad_id


    def causal_mask(self, y):
        sz = y.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)