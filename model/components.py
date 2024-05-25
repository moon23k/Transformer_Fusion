import torch, copy, math
import torch.nn as nn
import torch.nn.functional as F




def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



class Mapping(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_ratio=0.1):
        super(Mapping, self).__init__()            
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_ratio)        

    def forward(self, x):
        return self.dropout(self.linear(x))



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


#TBD
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.mha = nn.MultiheadAttention(
            config.hidden_dim, 
            config.n_heads, 
            batch_first=True
        )

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        x = self.norm(x)
        x = self.mha(query=q)[0]
        return self.dropout(x)
